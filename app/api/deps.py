"""Dependency'lar: autentifikatsiya va rate-limit."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

from fastapi import Request
from fastapi.security import APIKeyHeader

from app.config import get_settings
from app.envelope import ApiError, ErrorCode
from app.services.cache import cache

_settings = get_settings()

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


async def authenticate(request: Request) -> Principal:
    """`X-API-Key` yoki ishonchli IP bo'yicha kirish huquqini tekshiradi."""
    api_key = request.headers.get("x-api-key") or ""
    ip = client_ip(request)

    if api_key:
        record = await cache.lookup_key(api_key)
        if record is None:
            # Redis ishlamayotgan bo'lsa ham kalitni tasdiqlay olmaymiz.
            # Auth — fail-CLOSED: shubha bo'lsa kirishga ruxsat berilmaydi.
            raise ApiError(
                ErrorCode.UNAUTHORIZED,
                "API kaliti noto'g'ri yoki bekor qilingan",
                status_code=401,
            )
        if record.get("revoked"):
            raise ApiError(
                ErrorCode.UNAUTHORIZED, "API kaliti bekor qilingan", status_code=401
            )
        return Principal(
            kind="key",
            id=record.get("id") or cache.hash_key(api_key)[:12],
            rate_limit=int(record.get("rate_limit") or _settings.rate_limit_per_minute),
        )

    if _is_trusted(ip):
        return Principal(kind="trusted_ip", id=ip, rate_limit=_settings.rate_limit_per_minute)

    if _settings.public_mode:
        return Principal(kind="public", id=ip, rate_limit=_settings.rate_limit_per_minute)

    raise ApiError(
        ErrorCode.UNAUTHORIZED,
        "X-API-Key sarlavhasi talab qilinadi",
        status_code=401,
        headers={"WWW-Authenticate": "ApiKey"},
    )


async def enforce_rate_limit(request: Request, principal: Principal) -> dict[str, str]:
    """Limitni tekshiradi va javob uchun `X-RateLimit-*` sarlavhalarini qaytaradi."""
    limit = principal.rate_limit
    if limit <= 0:  # 0 yoki manfiy => cheksiz
        return {}

    subject = f"{principal.kind}:{principal.id}"
    current, ttl = await cache.hit_rate_limit(subject)

    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(max(0, limit - current)),
        "X-RateLimit-Reset": str(ttl),
    }
    if current > limit:
        raise ApiError(
            ErrorCode.RATE_LIMITED,
            f"So'rovlar limiti oshdi: daqiqasiga {limit} ta",
            status_code=429,
            details={"limit_per_minute": limit, "retry_after_s": ttl},
            headers={**headers, "Retry-After": str(ttl)},
        )
    return headers
