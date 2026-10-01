"""URL'dan rasm yuklash — SSRF himoyasi va qattiq limitlar bilan.

Himoya modeli:
  1. Faqat http/https sxemalari.
  2. DNS o'zimiz hal qilamiz va har bir IP `netguard` orqali tekshiriladi.
  3. So'rov TEKSHIRILGAN IP ga yuboriladi (`Host` va SNI asl domen bilan) —
     shunda DNS rebinding ishlamaydi: tekshirgan manzil bilan ulanadigan
     manzil kafolatli bir xil bo'ladi.
  4. Redirect'lar avtomatik kuzatilmaydi; har bir qadam qaytadan tekshiriladi.
  5. Yuklash oqim (stream) bilan, HAQIQIY o'qilgan baytlar bo'yicha uziladi —
     yolg'on `Content-Length` ga ishonilmaydi.
"""

from __future__ import annotations

import asyncio
import socket
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.core import netguard
from app.envelope import ApiError, ErrorCode

_settings = get_settings()

_USER_AGENT = "Mozilla/5.0 (compatible; media-fetcher/1.0)"


def _forbidden(url: str, reason: str) -> ApiError:
    return ApiError(
        ErrorCode.FORBIDDEN_TARGET,
        f"Bu manzilga so'rov yuborish taqiqlangan: {reason}",
        status_code=403,
        details={"url": url},
    )


def _prepare(url: str) -> tuple[httpx.URL, dict[str, str], dict[str, str]]:
    """URL'ni tekshirib, IP'ga bog'langan so'rov qismlarini tayyorlaydi.

    Qaytadi: `(ip_url, headers, extensions)`.
    """
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme
        host = parsed.hostname
        explicit_port = parsed.port
    except ValueError as exc:
        # `urlparse` chegaradan tashqari port yoki yopilmagan `[` uchun
        # `ValueError` ko'taradi — bu mijozning xatosi, 500 emas.
        raise ApiError(
            ErrorCode.INVALID_URL, f"URL yaroqsiz: {exc}", status_code=400
        ) from exc
    if scheme not in ("http", "https"):
        raise ApiError(
            ErrorCode.INVALID_URL,
            "Faqat http va https sxemalari qo'llab-quvvatlanadi",
            status_code=400,
            details={"scheme": scheme or None},
        )
    if not host:
        raise ApiError(
            ErrorCode.INVALID_URL, "URL da host ko'rsatilmagan", status_code=400
        )

    port = explicit_port or (443 if scheme == "https" else 80)

    try:
        candidates = netguard.resolve_all(host, port)
    except socket.gaierror as exc:
        raise ApiError(
            ErrorCode.FETCH_FAILED,
            f"Domen nomini aniqlab bo'lmadi: {host}",
            status_code=502,
        ) from exc
    except UnicodeError as exc:
        # Juda uzun IDNA yorlig'i `gaierror` emas, `UnicodeError` beradi.
        raise ApiError(
            ErrorCode.INVALID_URL, f"Domen nomi yaroqsiz: {host[:80]}", status_code=400
        ) from exc

    if _settings.allow_private_targets:
        chosen = candidates[0]
    else:
        picked = netguard.pick_safe_ip(candidates)
        if picked is None:
            raise _forbidden(url, "ichki yoki maxsus IP manzil")
        chosen = picked

    _family, ip = chosen

    # IPv6 URL ichida kvadrat qavsda yozilishi shart.
    ip_host = f"[{ip}]" if ":" in ip else ip
    netloc = f"{ip_host}:{port}"
    ip_url = httpx.URL(
        urlunparse(
            (scheme, netloc, parsed.path or "/", parsed.params, parsed.query, "")
        )
    )

    # `Host` — asl domen (virtual-host'lar to'g'ri ishlashi uchun).
    host_header = f"{host}:{port}" if explicit_port else host
    headers = {
        "Host": host_header,
        "User-Agent": _USER_AGENT,
        "Accept": "image/*,*/*;q=0.8",
        "Accept-Encoding": "identity",
    }
    # `sni_hostname` — httpcore kengaytmasi. U ham SNI, ham sertifikatni
    # tekshirish uchun ishlatiladi, ya'ni IP ga ulansak ham TLS to'liq
    # va to'g'ri domen bo'yicha tasdiqlanadi.
    extensions = {"sni_hostname": host} if scheme == "https" else {}
    return ip_url, headers, extensions


_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def get_client() -> httpx.AsyncClient:
    """Bitta umumiy mijoz.

    Ilgari har bir yuklashda yangi `AsyncClient` yaratilardi, ya'ni
    `max_connections` limiti HAR SO'ROV uchun alohida edi va xost bo'ylab
    ochiq ulanishlar soni cheklanmagan bo'lib qolardi.
    """
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        _settings.fetch_read_timeout,
                        connect=_settings.fetch_connect_timeout,
                        read=_settings.fetch_read_timeout,
                    ),
                    follow_redirects=False,  # har bir qadamni o'zimiz tekshiramiz
                    max_redirects=0,
                    limits=httpx.Limits(
                        max_connections=_settings.fetch_max_connections,
                        max_keepalive_connections=0,
                    ),
                )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def fetch_image(url: str) -> bytes:
    """URL'dan rasm baytlarini yuklaydi."""
    try:
        async with asyncio.timeout(_settings.fetch_total_timeout):
            return await _fetch(url)
    except TimeoutError as exc:
        # `httpx.Timeout` ning birinchi argumenti UMUMIY muddat EMAS — u
        # faqat o'rnatilmagan turlar uchun sukut qiymati. Sekin-drip
        # beruvchi manba `read=10s` ni har chunkda qayta boshlab, so'rovni
        # cheksiz ushlab turardi. Haqiqiy muddat shu yerda qo'yiladi.
        raise ApiError(
            ErrorCode.FETCH_TIMEOUT,
            "URL'dan yuklash vaqti tugadi",
            status_code=504,
            details={"timeout_s": _settings.fetch_total_timeout},
        ) from exc


async def _fetch(url: str) -> bytes:
    limit = _settings.max_image_bytes
    client = await get_client()

    current = url
    for _hop in range(_settings.fetch_max_redirects + 1):
        # DNS — bloklovchi libc chaqiruvi. Ilgari u to'g'ridan-to'g'ri
        # event loop ichida bajarilardi va javob bermaydigan domen butun
        # worker'ni soniyalarga to'xtatardi.
        ip_url, headers, extensions = await run_in_threadpool(_prepare, current)
        request = client.build_request(
            "GET", ip_url, headers=headers, extensions=extensions
        )
        try:
            response = await client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise ApiError(
                ErrorCode.FETCH_TIMEOUT,
                "URL'dan yuklash vaqti tugadi",
                status_code=504,
                details={"timeout_s": _settings.fetch_total_timeout},
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(
                ErrorCode.FETCH_FAILED,
                f"URL'ga ulanib bo'lmadi: {type(exc).__name__}",
                status_code=502,
            ) from exc

        try:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ApiError(
                        ErrorCode.FETCH_FAILED,
                        "Redirect javobida Location sarlavhasi yo'q",
                        status_code=502,
                    )
                # Nisbiy Location'ni ASL url'ga nisbatan hisoblaymiz
                # (IP ga bog'langan url'ga emas — aks holda keyingi
                # qadamda domen o'rniga IP tekshirilardi).
                current = str(httpx.URL(current).join(location))
                continue

            if response.status_code != 200:
                raise ApiError(
                    ErrorCode.FETCH_FAILED,
                    f"URL {response.status_code} javob qaytardi",
                    status_code=502,
                    details={"upstream_status": response.status_code},
                )

            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise ApiError(
                    ErrorCode.IMAGE_TOO_LARGE,
                    f"Rasm hajmi {limit // (1024 * 1024)} MB dan oshmasligi kerak",
                    status_code=413,
                    details={"size_bytes": int(declared), "limit_bytes": limit},
                )

            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes(65536):
                total += len(chunk)
                if total > limit:
                    raise ApiError(
                        ErrorCode.IMAGE_TOO_LARGE,
                        f"Rasm hajmi {limit // (1024 * 1024)} MB dan oshmasligi kerak",
                        status_code=413,
                        details={"limit_bytes": limit},
                    )
                chunks.append(chunk)
        finally:
            await response.aclose()

        data = b"".join(chunks)
        if not data:
            raise ApiError(
                ErrorCode.FETCH_FAILED, "URL bo'sh javob qaytardi", status_code=502
            )
        return data

    raise ApiError(
        ErrorCode.FETCH_FAILED,
        f"Redirect'lar soni {_settings.fetch_max_redirects} dan oshdi",
        status_code=502,
    )
