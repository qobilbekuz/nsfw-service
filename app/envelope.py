"""Standart JSON javob envelope'i va xato kodlari.

Servisdagi HAR BIR javob — muvaffaqiyatli yoki xato — shu shaklda chiqadi:

    {"success": bool, "request_id": str, "took_ms": int,
     "data": {...} | null, "error": {...} | null}

`success` va `error` hech qachon birga to'ldirilmaydi, shuning uchun mijoz
faqat `success` ni tekshirsa yetarli.
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
) -> dict[str, Any]:
    return {
        "success": True,
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
) -> dict[str, Any]:
    return {
        "success": False,
        "request_id": request_id,
        "took_ms": took_ms,
        "data": None,
        "error": {"code": code, "message": message, "details": details},
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
        content=error_body(code, message, request_id, _took_ms(request), details),
        headers=out_headers,
    )
