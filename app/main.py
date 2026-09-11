"""NSFW Detection API — FastAPI ilovasi.

Ishga tushirish (systemd orqali):
    uvicorn app.main:app --host 127.0.0.1 --port 8720 --root-path /nsfw
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.config import get_settings
from app.envelope import ApiError, ErrorCode, error_response, new_request_id
from app.services import pipeline
from app.services.cache import cache
from app.services import classifier as classifier_mod
from app.services.classifier import SafetyClassifier
from app.services.detector import BodyPartDetector

settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("nsfw")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modellarni bir marta yuklaydi — har so'rovda emas."""
    started = time.perf_counter()

    model_path, head = settings.classifier_path, None
    if settings.custom_head_path:
        # Nosozlik bo'lsa servisni tushirmaymiz — asl boshga qaytamiz.
        # Moderatsiya butunlay to'xtagandan ko'ra, biroz aniqroq bo'lmagan
        # model bilan ishlagan afzal. Log ogohlantiradi, `/v1/models` esa
        # qaysi bosh ishlayotganini ko'rsatadi.
        try:
            head = classifier_mod.load_custom_head(settings.custom_head_path)
            model_path = classifier_mod.ensure_feature_model(
                settings.classifier_path, settings.feature_model_path
            )
        except Exception:
            log.exception(
                "o'z boshingizni yuklab bo'lmadi (%s) — asl bosh ishlatiladi",
                settings.custom_head_file,
            )
            model_path, head = settings.classifier_path, None

    pipeline.engine.classifier = SafetyClassifier(
        model_path,
        intra_threads=settings.onnx_intra_threads,
        inter_threads=settings.onnx_inter_threads,
        custom_head=head,
    )
    if settings.detector_enabled:
        pipeline.engine.detector = BodyPartDetector(
            intra_threads=settings.onnx_intra_threads,
            inter_threads=settings.onnx_inter_threads,
        )

    await cache.connect()
    log.info(
        "modellar yuklandi (%.0f ms), redis=%s",
        (time.perf_counter() - started) * 1000,
        "ok" if cache.available else "yo'q",
    )
    try:
        yield
    finally:
        await cache.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "18+ (NSFW/NSFL) rasm aniqlovchi servis. Rasmni URL, server yo'li, "
        "base64 yoki fayl yuklash orqali qabul qiladi va natijani foizda qaytaradi."
    ),
    root_path=settings.root_path,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Har bir so'rovga `request_id` va vaqt hisoblagichini biriktiradi."""
    request.state.started_at = time.perf_counter()
    request.state.request_id = request.headers.get("x-request-id") or new_request_id()

    try:
        response = await call_next(request)
    except Exception:
        log.exception("ushlanmagan xato request_id=%s", request.state.request_id)
        return error_response(
            request,
            ErrorCode.INTERNAL_ERROR,
            "Ichki xato yuz berdi",
            status_code=500,
        )

    response.headers["X-Request-ID"] = request.state.request_id
    took_ms = int((time.perf_counter() - request.state.started_at) * 1000)

    if request.url.path.startswith("/v1/") and not request.url.path.endswith("/health"):
        principal = getattr(request.state, "principal", None)
        # DIQQAT: URL, fayl yo'li va rasm mazmuni ATAYLAB loglanmaydi —
        # bu moderatsiya servisi, loglar maxfiy ma'lumot to'plamiga
        # aylanmasligi kerak.
        log.info(
            "%s %s status=%s took_ms=%s key=%s rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            took_ms,
            f"{principal.kind}:{principal.id}" if principal else "-",
            request.state.request_id,
        )
    return response


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    return error_response(
        request,
        exc.code,
        exc.message,
        status_code=exc.status_code,
        details=exc.details,
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    message = first.get("msg", "so'rov formati noto'g'ri")
    return error_response(
        request,
        ErrorCode.INVALID_REQUEST,
        f"{location}: {message}" if location else message,
        status_code=400,
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """404/405 kabi framework xatolarini ham bir xil envelope'ga soladi."""
    code = {
        401: ErrorCode.UNAUTHORIZED,
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        413: ErrorCode.IMAGE_TOO_LARGE,
        415: ErrorCode.UNSUPPORTED_MEDIA_TYPE,
    }.get(exc.status_code, ErrorCode.INVALID_REQUEST if exc.status_code < 500 else ErrorCode.INTERNAL_ERROR)
    return error_response(
        request,
        code,
        str(exc.detail) if exc.detail else "So'rov bajarilmadi",
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


app.include_router(router)


@app.get("/", include_in_schema=False)
async def index(request: Request) -> JSONResponse:
    from app.envelope import success_response

    return success_response(
        request,
        {
            "service": settings.app_name,
            "version": settings.version,
            "docs": f"{settings.root_path}/docs",
            "endpoints": [
                "POST /v1/analyze",
                "GET  /v1/analyze?url=",
                "POST /v1/analyze/batch",
                "GET  /v1/health",
                "GET  /v1/models",
            ],
        },
    )
