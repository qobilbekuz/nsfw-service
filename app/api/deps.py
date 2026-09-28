"""Dependency'lar: autentifikatsiya va rate-limit."""

from __future__ import annotations

import hmac
import ipaddress
import logging
from dataclasses import dataclass

from fastapi import Request
from fastapi.security import APIKeyHeader

from app.config import get_settings
from app.envelope import ApiError, ErrorCode
from app.services.cache import FOUND, UNAVAILABLE, cache

_settings = get_settings()
log = logging.getLogger("nsfw.auth")

#: Faqat OpenAPI hujjati uchun — Swagger UI'da "Authorize" tugmasi shundan
#: paydo bo'ladi. Tekshiruvni `authenticate()` o'zi qiladi, shu sababli
#: `auto_error=False`: bu sxema hech qachon o'zi 401 qaytarmaydi va
#: ishonchli IP / public rejimni buzmaydi.
api_key_scheme = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="`scripts/apikey.py create` orqali olingan kalit (`nsfw_...`).",
)


@dataclass(slots=True)
class Principal:
    """So'rovni kim yuborganini bildiruvchi identifikator."""

    kind: str  # "key" | "trusted_ip" | "public"
    id: str
    rate_limit: int


def client_ip(request: Request) -> str:
    """Haqiqiy mijoz IP'si.

    uvicorn `--forwarded-allow-ips=127.0.0.1` bilan ishga tushadi, ya'ni
    `X-Forwarded-For` ni faqat nginx qo'ygan bo'lsa hisobga oladi va
    `request.client.host` ni allaqachon to'g'irlab beradi.
    """
    return request.client.host if request.client else "unknown"


def _is_trusted(ip: str) -> bool:
    for entry in _settings.trusted_ips:
        try:
            if "/" in entry:
                if ipaddress.ip_address(ip) in ipaddress.ip_network(entry, strict=False):
                    return True
            elif ip == entry:
                return True
        except ValueError:
            continue
    return False


def _bootstrap_lookup(digest: str) -> dict | None:
    """`.env` dagi BOOTSTRAP_KEYS (`id:sha256[:rate_limit]`) orasidan qidiradi.

    Solishtirish `hmac.compare_digest` bilan — vaqt bo'yicha sizib chiqmasin.
    Yaroqsiz yozuv jimgina o'tkazib yuboriladi (servis tushmasin), lekin logda
    ogohlantirish qoladi.
    """
    for entry in _settings.bootstrap_keys:
        parts = entry.split(":")
        if len(parts) < 2 or len(parts[1]) != 64:
            log.warning("BOOTSTRAP_KEYS yozuvi yaroqsiz: %r", parts[0] if parts else entry)
            continue
        if hmac.compare_digest(parts[1].lower(), digest):
            rate = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
            # `0` — cheksiz; `None` — sozlanmagan, sukut limiti
            return {"id": parts[0], "rate_limit": rate, "source": "bootstrap"}
    return None


async def authenticate(request: Request) -> Principal:
    """`X-API-Key` yoki ishonchli IP bo'yicha kirish huquqini tekshiradi."""
    api_key = request.headers.get("x-api-key") or ""
    ip = client_ip(request)

    if api_key:
        state, record = await cache.lookup_key(api_key)
        if state != FOUND:
            # Auth — fail-CLOSED: noma'lum kalit HECH QACHON o'tmaydi. Faqat
            # `.env` da dayjesti yozilgan zaxira kalitlar Redis o'chsa yoki
            # tozalansa ham ishlaydi (ilgari bunday holatda HAMMA mijoz 401
            # olardi — kalitlar faqat Redis'da, AOF o'chiq).
            record = _bootstrap_lookup(cache.hash_key(api_key))
            if record is None:
                raise ApiError(
                    ErrorCode.UNAUTHORIZED,
                    "API kaliti noto'g'ri yoki bekor qilingan",
                    status_code=401,
                )
            if state == UNAVAILABLE:
                log.warning("Redis yo'q — zaxira kalit ishlatildi: %s", record["id"])
        if record.get("revoked"):
            raise ApiError(
                ErrorCode.UNAUTHORIZED, "API kaliti bekor qilingan", status_code=401
            )
        # `or` bu yerda ISHLAMAYDI: `0` ("cheksiz" deb hujjatlashtirilgan)
        # falsy bo'lgani uchun jimgina sukut limitiga aylanardi.
        configured = record.get("rate_limit")
        return Principal(
            kind="key",
            id=record.get("id") or cache.hash_key(api_key)[:12],
            rate_limit=(
                int(configured)
                if isinstance(configured, int | float | str)
                and str(configured).lstrip("-").isdigit()
                else _settings.rate_limit_per_minute
            ),
        )

    if _is_trusted(ip):
        trusted_limit = _settings.trusted_rate_limit_per_minute
        return Principal(
            kind="trusted_ip",
            id=ip,
            rate_limit=_settings.rate_limit_per_minute if trusted_limit is None else trusted_limit,
        )

    if _settings.public_mode:
        return Principal(kind="public", id=ip, rate_limit=_settings.rate_limit_per_minute)

    raise ApiError(
        ErrorCode.UNAUTHORIZED,
        "X-API-Key sarlavhasi talab qilinadi",
        status_code=401,
        headers={"WWW-Authenticate": "ApiKey"},
    )


async def enforce_rate_limit(
    request: Request, principal: Principal, cost: int = 1
) -> dict[str, str]:
    """Limitni tekshiradi va javob uchun `X-RateLimit-*` sarlavhalarini qaytaradi.

    `cost` — so'rov "narxi" (tahlil qilinadigan rasmlar soni). Ilgari batch
    20 rasm bo'lsa ham 1 birlik yerdi — limit 20 barobar aylanib o'tilardi.
    """
    limit = principal.rate_limit
    if limit <= 0:  # 0 yoki manfiy => cheksiz
        return {}

    subject = f"{principal.kind}:{principal.id}"
    current, ttl = await cache.hit_rate_limit(subject, cost=cost)

    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(max(0, limit - current)),
        "X-RateLimit-Reset": str(ttl),
    }
    if current > limit:
        # Rad etilgan ish uchun kvota yechilmaydi: aks holda limitga yetgan
        # mijoz bitta 20 ta lik batch bilan o'zini oynaning oxirigacha
        # to'liq bloklab qo'yardi.
        await cache.refund_rate_limit(subject, cost=cost)
        headers["X-RateLimit-Remaining"] = str(max(0, limit - (current - cost)))
        raise ApiError(
            ErrorCode.RATE_LIMITED,
            f"So'rovlar limiti oshdi: daqiqasiga {limit} ta",
            status_code=429,
            details={"limit_per_minute": limit, "retry_after_s": ttl, "cost": cost},
            headers={**headers, "Retry-After": str(ttl)},
        )
    return headers
