"""Standart JSON javob envelope'i va xato kodlari.

Servisdagi HAR BIR javob — muvaffaqiyatli yoki xato — shu shaklda chiqadi:

    {"success": bool, "ok": bool, "status_code": int, "request_id": str,
     "took_ms": int, "data": {...} | null,
     "error": {"code", "message", "messages": {uz,ru,en}, "details"} | null}

`success` va `error` hech qachon birga to'ldirilmaydi, shuning uchun mijoz
faqat `success` ni tekshirsa yetarli.

`ok` / `status_code` / `error.messages` (2026-09-25) — api.qobilbek.dev dagi
PHP API'lar (tiktok, pinterest, likee, ...) bilan bir xil maydonlar. Ular
QO'SHIMCHA: eski mijozlar o'qiydigan maydonlar o'zgarmadi. `error.message` —
aniq (kontekstli) o'zbekcha matn, `error.messages` — kod bo'yicha 3 tilda.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class ErrorCode:
    """`error.code` uchun barqaror (o'zgarmaydigan) qiymatlar."""

    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_URL = "INVALID_URL"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN_TARGET = "FORBIDDEN_TARGET"
    PATH_NOT_ALLOWED = "PATH_NOT_ALLOWED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_TOO_LARGE_PIXELS = "IMAGE_TOO_LARGE_PIXELS"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    DECODE_FAILED = "DECODE_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    FETCH_FAILED = "FETCH_FAILED"
    FETCH_TIMEOUT = "FETCH_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"


#: Kod bo'yicha 3 tildagi umumiy xabar (PHP API'lardagi `message{uz,ru,en}`).
MESSAGES: dict[str, dict[str, str]] = {
    ErrorCode.INVALID_REQUEST: {"uz": "So'rov noto'g'ri tuzilgan.", "ru": "Некорректный запрос.", "en": "Invalid request."},
    ErrorCode.INVALID_URL: {"uz": "URL noto'g'ri.", "ru": "Некорректный URL.", "en": "Invalid URL."},
    ErrorCode.UNAUTHORIZED: {"uz": "API kaliti noto'g'ri yoki berilmagan.", "ru": "Неверный или отсутствующий API-ключ.", "en": "Missing or invalid API key."},
    ErrorCode.FORBIDDEN_TARGET: {"uz": "Bu manzilga so'rov yuborish taqiqlangan.", "ru": "Запрос к этому адресу запрещён.", "en": "Requests to this address are not allowed."},
    ErrorCode.PATH_NOT_ALLOWED: {"uz": "Fayl topilmadi yoki ruxsat yo'q.", "ru": "Файл не найден или доступ запрещён.", "en": "File not found or not allowed."},
    ErrorCode.FILE_NOT_FOUND: {"uz": "Fayl topilmadi.", "ru": "Файл не найден.", "en": "File not found."},
    ErrorCode.IMAGE_TOO_LARGE: {"uz": "Rasm hajmi juda katta.", "ru": "Изображение слишком большое.", "en": "Image is too large."},
    ErrorCode.IMAGE_TOO_LARGE_PIXELS: {"uz": "Rasm o'lchami juda katta.", "ru": "Слишком большое разрешение изображения.", "en": "Image resolution is too large."},
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: {"uz": "Rasm formati qo'llab-quvvatlanmaydi.", "ru": "Формат изображения не поддерживается.", "en": "Unsupported image format."},
    ErrorCode.DECODE_FAILED: {"uz": "Rasmni o'qib bo'lmadi.", "ru": "Не удалось прочитать изображение.", "en": "Could not decode the image."},
    ErrorCode.RATE_LIMITED: {"uz": "So'rovlar limiti oshdi, keyinroq urinib ko'ring.", "ru": "Превышен лимит запросов, повторите позже.", "en": "Rate limit exceeded, please try again later."},
    ErrorCode.FETCH_FAILED: {"uz": "Rasmni URL'dan yuklab bo'lmadi.", "ru": "Не удалось загрузить изображение по URL.", "en": "Could not fetch the image from the URL."},
    ErrorCode.FETCH_TIMEOUT: {"uz": "URL'dan yuklash vaqti tugadi.", "ru": "Истекло время загрузки по URL.", "en": "Fetching the URL timed out."},
    ErrorCode.INTERNAL_ERROR: {"uz": "Ichki xato yuz berdi.", "ru": "Внутренняя ошибка.", "en": "Internal error."},
    ErrorCode.SERVICE_UNAVAILABLE: {"uz": "Servis vaqtincha ishlamayapti.", "ru": "Сервис временно недоступен.", "en": "Service temporarily unavailable."},
    ErrorCode.NOT_FOUND: {"uz": "Manzil topilmadi.", "ru": "Не найдено.", "en": "Not found."},
    ErrorCode.METHOD_NOT_ALLOWED: {"uz": "Bu metod ruxsat etilmagan.", "ru": "Метод не разрешён.", "en": "Method not allowed."},
}


class ApiError(Exception):
    """Boshqariladigan xato — main.py dagi handler uni envelope'ga aylantiradi."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        self.headers = headers


def new_request_id() -> str:
    return f"r_{secrets.token_hex(10)}"


def _took_ms(request: Request | None) -> int:
    started = getattr(request.state, "started_at", None) if request else None
    if started is None:
        return 0
    return int((time.perf_counter() - started) * 1000)


def success_body(
    data: Any,
    request_id: str,
    took_ms: int,
    status_code: int = 200,
) -> dict[str, Any]:
    return {
        "success": True,
        "ok": True,
        "status_code": status_code,
        "request_id": request_id,
        "took_ms": took_ms,
        "data": data,
        "error": None,
    }


def error_body(
    code: str,
    message: str,
    request_id: str,
    took_ms: int,
    details: dict[str, Any] | None = None,
    status_code: int = 400,
) -> dict[str, Any]:
    return {
        "success": False,
        "ok": False,
        "status_code": status_code,
        "request_id": request_id,
        "took_ms": took_ms,
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "messages": MESSAGES.get(code, MESSAGES[ErrorCode.INTERNAL_ERROR] if status_code >= 500 else MESSAGES[ErrorCode.INVALID_REQUEST]),
            "details": details,
        },
    }


def success_response(request: Request, data: Any) -> JSONResponse:
    request_id = getattr(request.state, "request_id", new_request_id())
    return JSONResponse(
        status_code=200,
        content=success_body(data, request_id, _took_ms(request)),
        headers={"X-Request-ID": request_id},
    )


def error_response(
    request: Request | None,
    code: str,
    message: str,
    status_code: int = 400,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = (
        getattr(request.state, "request_id", None) if request else None
    ) or new_request_id()
    out_headers = {"X-Request-ID": request_id}
    if headers:
        out_headers.update(headers)
    return JSONResponse(
        status_code=status_code,
        content=error_body(code, message, request_id, _took_ms(request), details, status_code),
        headers=out_headers,
    )
