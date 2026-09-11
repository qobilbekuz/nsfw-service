"""Rasm baytlarini turli kirish rejimlaridan olish: path va base64."""

from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path

from app.config import get_settings
from app.envelope import ApiError, ErrorCode

_settings = get_settings()

_DATA_URI = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


def load_from_path(raw_path: str) -> bytes:
    """Serverdagi fayldan o'qiydi — faqat oq ro'yxatdagi kataloglar ichidan.

    Himoya: `resolve()` symlink'larni ochadi va `..` ni yig'ishtiradi, shundan
    KEYIN ildizga tegishlilik tekshiriladi. Ya'ni `../../etc/passwd` ham,
    tashqariga ishora qiluvchi symlink ham o'tmaydi.
    """
    roots = _settings.allowed_path_roots
    if not roots:
        raise ApiError(
            ErrorCode.PATH_NOT_ALLOWED,
            "Path rejimi o'chirilgan (ALLOWED_PATH_ROOTS sozlanmagan)",
            status_code=403,
        )

    if "\x00" in raw_path:
        raise ApiError(
            ErrorCode.PATH_NOT_ALLOWED, "Yo'lda yaroqsiz belgi bor", status_code=403
        )

    candidate = Path(raw_path)
    # Nisbiy yo'l birinchi ildizga nisbatan hisoblanadi.
    resolved_candidates = (
        [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
    )

    for item in resolved_candidates:
        try:
            resolved = item.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not any(resolved.is_relative_to(root) for root in roots):
            continue
        if not resolved.is_file():
            continue
        try:
            size = resolved.stat().st_size
        except OSError:
            continue
        if size > _settings.max_image_bytes:
            raise ApiError(
                ErrorCode.IMAGE_TOO_LARGE,
                f"Fayl hajmi {_settings.max_image_bytes // (1024 * 1024)} MB dan katta",
                status_code=413,
                details={"size_bytes": size},
            )
        try:
            return resolved.read_bytes()
        except OSError as exc:
            raise ApiError(
                ErrorCode.FILE_NOT_FOUND,
                "Faylni o'qib bo'lmadi",
                status_code=404,
            ) from exc

    # Ruxsat etilgan ildiz ichida topilmadi. Fayl bor-yo'qligini oshkor
    # qilmaslik uchun "topilmadi" va "ruxsat yo'q" ni farqlamaymiz — aks holda
    # bu API fayl tizimini skanerlash vositasiga aylanadi.
    raise ApiError(
        ErrorCode.PATH_NOT_ALLOWED,
        "Fayl topilmadi yoki ruxsat etilgan katalogdan tashqarida",
        status_code=403,
        details={"allowed_roots": [str(r) for r in roots]},
    )


def load_from_base64(raw: str) -> bytes:
    """Base64 satrni (data-URI prefiksi bilan yoki busiz) baytga aylantiradi."""
    payload = _DATA_URI.sub("", raw.strip())
    # Bazi mijozlar URL-safe alifboda yuboradi.
    payload = payload.replace("-", "+").replace("_", "/")
    padding = len(payload) % 4
    if padding:
        payload += "=" * (4 - padding)

    # Dekodlashdan OLDIN taxminiy hajmni tekshiramiz: base64 ~4/3 marta katta.
    approx = len(payload) * 3 // 4
    if approx > _settings.max_image_bytes:
        raise ApiError(
            ErrorCode.IMAGE_TOO_LARGE,
            f"Rasm hajmi {_settings.max_image_bytes // (1024 * 1024)} MB dan oshmasligi kerak",
            status_code=413,
            details={"approx_bytes": approx},
        )

    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ApiError(
            ErrorCode.INVALID_REQUEST,
            "image_base64 yaroqli base64 satr emas",
            status_code=400,
        ) from exc
