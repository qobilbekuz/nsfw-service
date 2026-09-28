"""Rasm dekodlash — dekompressiya bombasi va noto'g'ri formatlardan himoya bilan.

Rasm HECH QACHON diskka yozilmaydi: barcha ish xotirada bajariladi.
"""

from __future__ import annotations

import hashlib
import io
import threading

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import get_settings
from app.envelope import ApiError, ErrorCode
from app.schemas import ImageInfo

_settings = get_settings()

# Dekompressiya bombasi ("zip bomb" ning rasm varianti): 200x200 baytlik PNG
# ochilganda 100 megapiksel bo'lib xotirani to'ldirishi mumkin. Pillow bunga
# ogohlantirish beradi, lekin biz qattiq chegara qo'yamiz.
Image.MAX_IMAGE_PIXELS = _settings.max_image_pixels

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "MPO"}

# Fayl imzolari (magic bytes). Content-Type sarlavhasiga ishonib bo'lmaydi —
# uni yuboruvchi taraf xohlaganini yozadi.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"BM", "BMP"),
)

# Decoding a 50 MP image materialises ~150 MB before anything can shrink it.
# The guard bounds how many of those can exist at once inside one worker.
_decode_slots = threading.BoundedSemaphore(max(1, _settings.max_concurrent_decodes))


def sniff_format(data: bytes) -> str | None:
    """Bayt imzosi bo'yicha formatni aniqlash (WEBP alohida — RIFF konteyneri)."""
    for magic, fmt in _MAGIC:
        if data.startswith(magic):
            return fmt
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def looks_like_image(data: bytes) -> bool:
    return sniff_format(data) is not None


def check_size(size_bytes: int) -> None:
    limit = _settings.max_image_bytes
    if size_bytes > limit:
        raise ApiError(
            ErrorCode.IMAGE_TOO_LARGE,
            f"Rasm hajmi {limit // (1024 * 1024)} MB dan oshmasligi kerak",
            status_code=413,
            details={"size_bytes": size_bytes, "limit_bytes": limit},
        )


def _work_size(width: int, height: int) -> tuple[int, int] | None:
    """Target size that keeps the aspect ratio under `max_working_pixels`."""
    budget = _settings.max_working_pixels
    if budget <= 0 or width * height <= budget:
        return None
    factor = (budget / (width * height)) ** 0.5
    return max(1, int(width * factor)), max(1, int(height * factor))


def _decode_locked(data: bytes) -> tuple[Image.Image, str, int, int]:
    img = Image.open(io.BytesIO(data))

    # Piksel chegarasini O'ZIMIZ tekshiramiz. Pillow'ning o'z himoyasi
    # MAX_IMAGE_PIXELS da faqat ogohlantirish beradi va xatoni ikki
    # baravar chegaradan keyin ko'taradi — ya'ni 50 MP limit qo'yilgani
    # bilan 99 MP rasm jimgina o'tib ketardi. `Image.open` lazy ishlaydi,
    # shuning uchun `size` piksellarni ajratmasdan oldin ma'lum bo'ladi.
    if img.width * img.height > _settings.max_image_pixels:
        raise ApiError(
            ErrorCode.IMAGE_TOO_LARGE_PIXELS,
            "Rasm o'lchami juda katta (dekompressiya bombasi himoyasi)",
            status_code=422,
            details={
                "pixels": img.width * img.height,
                "max_pixels": _settings.max_image_pixels,
            },
        )

    fmt = (img.format or sniff_format(data) or "UNKNOWN").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ApiError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            f"Format qo'llab-quvvatlanmaydi: {fmt}",
            status_code=415,
        )

    full_width, full_height = img.width, img.height
    target = _work_size(full_width, full_height)
    # JPEG can be decoded straight at a reduced scale, so the full-size
    # buffer is never allocated. Other formats must be decoded in full.
    if target is not None:
        img.draft("RGB", target)

    # Pillow lazy ishlaydi — piksellarni shu yerda majburan o'qiymiz, shunda
    # buzilgan fayl inference paytida emas, aynan shu yerda xato beradi.
    img.load()
    loaded = img.size
    # EXIF orientatsiyasini qo'llaymiz: telefonda olingan rasmlar aks holda
    # yonboshiga yotgan holda modelga tushadi.
    img = ImageOps.exif_transpose(img) or img
    if img.size == (loaded[1], loaded[0]) and loaded[0] != loaded[1]:
        full_width, full_height = full_height, full_width
    if img.mode != "RGB":
        img = img.convert("RGB")
    target = _work_size(img.width, img.height)
    if target is not None:
        img = img.resize(target, Image.Resampling.BILINEAR)
    return img, fmt, full_width, full_height


def decode(
    data: bytes, sha256: str | None = None
) -> tuple[Image.Image, ImageInfo, float]:
    """Baytlarni RGB rasmga aylantiradi va meta ma'lumot qaytaradi.

    Uchinchi qiymat — ishchi rasmdan ASL o'lchamga o'tish koeffitsienti:
    katta rasm modelga kichraytirib beriladi, lekin `ImageInfo` va
    topilma ramkalari mijozga asl koordinatalarda ko'rsatiladi.

    Animatsiyali GIF/WEBP uchun faqat birinchi kadr olinadi.
    """
    check_size(len(data))

    if not looks_like_image(data):
        raise ApiError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            "Fayl qo'llab-quvvatlanadigan rasm formati emas (JPEG, PNG, WEBP, GIF, BMP)",
            status_code=415,
        )

    with _decode_slots:
        try:
            rgb, fmt, full_width, full_height = _decode_locked(data)
        except ApiError:
            raise
        except Image.DecompressionBombError as exc:
            raise ApiError(
                ErrorCode.IMAGE_TOO_LARGE_PIXELS,
                "Rasm o'lchami juda katta (dekompressiya bombasi himoyasi)",
                status_code=422,
                details={"max_pixels": _settings.max_image_pixels},
            ) from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ApiError(
                ErrorCode.DECODE_FAILED,
                "Rasmni o'qib bo'lmadi — fayl buzilgan yoki rasm emas",
                status_code=422,
            ) from exc

    scale = (full_width / rgb.width) if rgb.width else 1.0

    info = ImageInfo(
        width=full_width,
        height=full_height,
        format=fmt,
        size_bytes=len(data),
        # Pipeline kesh kaliti uchun allaqachon hisoblagan — ikkinchi marta
        # 20 MB ni xeshlash shart emas.
        sha256=sha256 or hashlib.sha256(data).hexdigest(),
    )
    return rgb, info, scale


def to_bgr_array(img: Image.Image) -> np.ndarray:
    """NudeNet OpenCV kutadi — u BGR kanal tartibida ishlaydi."""
    return np.asarray(img, dtype=np.uint8)[:, :, ::-1].copy()
