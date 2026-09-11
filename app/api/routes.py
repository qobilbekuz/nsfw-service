"""API marshrutlari — barcha javoblar standart envelope orqali qaytadi."""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core import imaging
from app.envelope import ApiError, ErrorCode, error_body, success_response
from app.schemas import (
    AnalyzeRequest,
    BatchItem,
    BatchRequest,
    BatchResult,
    BatchResultItem,
    HealthResult,
    ModelsResult,
)
from app.api.deps import api_key_scheme, authenticate, enforce_rate_limit
from app.services import pipeline
from app.services.cache import cache

router = APIRouter(prefix="/v1")
_settings = get_settings()

#: Marshrutni "kalit talab qiladi" deb belgilaydi — bu faqat hujjatga ta'sir
#: qiladi, haqiqiy tekshiruv `_guard()` ichida qoladi.
_secured = [Depends(api_key_scheme)]

_STARTED_AT = time.time()


async def _guard(request: Request) -> dict[str, str]:
    """Auth + rate-limit. Javob sarlavhalarini qaytaradi."""
    principal = await authenticate(request)
    request.state.principal = principal
    return await enforce_rate_limit(request, principal)


def _with_headers(response: JSONResponse, headers: dict[str, str]) -> JSONResponse:
    for key, value in headers.items():
        response.headers[key] = value
    return response


# ---------------------------------------------------------------------------
#  Tahlil
# ---------------------------------------------------------------------------


@router.post("/analyze", summary="Bitta rasmni tahlil qilish", dependencies=_secured)
async def analyze(
    request: Request,
    file: UploadFile | None = File(
        default=None, description="multipart/form-data rejimida rasm fayli"
    ),
    options: str | None = Form(
        default=None,
        description='multipart rejimida ixtiyoriy JSON: {"detect":true,"cache":true}',
    ),
) -> JSONResponse:
    """To'rt kirish rejimi: `url`, `path`, `image_base64` (JSON) yoki `file` (multipart)."""
    rl_headers = await _guard(request)

    content_type = (request.headers.get("content-type") or "").lower()

    if file is not None or content_type.startswith("multipart/form-data"):
        if file is None:
            raise ApiError(
                ErrorCode.INVALID_REQUEST,
                "multipart so'rovida `file` maydoni topilmadi",
                status_code=400,
            )
        opts = _parse_options(options)
        started = time.perf_counter()
        data = await file.read()
        read_ms = int((time.perf_counter() - started) * 1000)
        imaging.check_size(len(data))
        result = await pipeline.analyze_bytes(
            data,
            detect=opts.detect,
            min_score=opts.min_detection_score,
            use_cache=opts.cache,
            fetch_ms=read_ms,
        )
    else:
        body = await _json_body(request)
        try:
            payload = AnalyzeRequest.model_validate(body)
        except ValueError as exc:
            raise ApiError(
                ErrorCode.INVALID_REQUEST, _first_error(exc), status_code=400
            ) from exc
        result = await pipeline.analyze_item(BatchItem(**payload.model_dump()))

    return _with_headers(
        success_response(request, result.model_dump(mode="json")), rl_headers
    )


@router.get("/analyze", summary="URL bo'yicha tez tahlil (GET)", dependencies=_secured)
async def analyze_get(
    request: Request,
    url: str,
    detect: bool = True,
    min_detection_score: float = 25.0,
    cache_enabled: bool = True,
) -> JSONResponse:
    rl_headers = await _guard(request)
    item = BatchItem(
        url=url,
        detect=detect,
        min_detection_score=min_detection_score,
        cache=cache_enabled,
    )
    result = await pipeline.analyze_item(item)
    return _with_headers(
        success_response(request, result.model_dump(mode="json")), rl_headers
    )


@router.post(
    "/analyze/batch",
    summary="Ko'p rasmni birdaniga tahlil qilish",
    dependencies=_secured,
)
async def analyze_batch(request: Request) -> JSONResponse:
    rl_headers = await _guard(request)
    body = await _json_body(request)
    try:
        payload = BatchRequest.model_validate(body)
    except ValueError as exc:
        raise ApiError(
            ErrorCode.INVALID_REQUEST, _first_error(exc), status_code=400
        ) from exc

    if len(payload.items) > _settings.max_batch_items:
        raise ApiError(
            ErrorCode.INVALID_REQUEST,
            f"Bir so'rovda maksimum {_settings.max_batch_items} ta rasm",
            status_code=400,
            details={"count": len(payload.items), "limit": _settings.max_batch_items},
        )

    semaphore = asyncio.Semaphore(_settings.batch_concurrency)

    async def run(index: int, item: BatchItem) -> BatchResultItem:
        async with semaphore:
            try:
                result = await pipeline.analyze_item(item)
                return BatchResultItem(
                    id=item.id, index=index, success=True, data=result
                )
            except ApiError as exc:
                return BatchResultItem(
                    id=item.id,
                    index=index,
                    success=False,
                    error={
                        "code": exc.code,
                        "message": exc.message,
                        "details": exc.details,
                    },
                )
            except Exception:  # noqa: BLE001 — bitta element butun batchni yiqitmasin
                return BatchResultItem(
                    id=item.id,
                    index=index,
                    success=False,
                    error={
                        "code": ErrorCode.INTERNAL_ERROR,
                        "message": "Ichki xato",
                        "details": None,
                    },
                )

    results = await asyncio.gather(
        *(run(i, item) for i, item in enumerate(payload.items))
    )
    succeeded = sum(1 for r in results if r.success)
    summary = BatchResult(
        count=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=list(results),
    )
    return _with_headers(
        success_response(request, summary.model_dump(mode="json")), rl_headers
    )


# ---------------------------------------------------------------------------
#  Xizmat endpointlari
# ---------------------------------------------------------------------------


@router.get("/health", summary="Servis holati (auth talab qilinmaydi)")
async def health(request: Request) -> JSONResponse:
    redis_state = "disabled"
    if _settings.cache_enabled:
        redis_state = "ok" if await cache.ping() else "down"

    models_loaded = pipeline.engine.ready
    result = HealthResult(
        status="ok" if models_loaded else "degraded",
        version=_settings.version,
        models_loaded=models_loaded,
        redis=redis_state,
        uptime_s=int(time.time() - _STARTED_AT),
    )
    return success_response(request, result.model_dump(mode="json"))


@router.get(
    "/models", summary="Yuklangan modellar va chegaralar", dependencies=_secured
)
async def models(request: Request) -> JSONResponse:
    await _guard(request)
    engine = pipeline.engine
    result = ModelsResult(
        classifier=engine.classifier.info() if engine.classifier else {},
        detector=engine.detector.info() if engine.detector else {"enabled": False},
        limits={
            "max_image_bytes": _settings.max_image_bytes,
            "max_image_pixels": _settings.max_image_pixels,
            "max_batch_items": _settings.max_batch_items,
            "rate_limit_per_minute": _settings.rate_limit_per_minute,
            "allowed_formats": sorted(imaging.ALLOWED_FORMATS - {"MPO"}),
        },
        thresholds={
            "nsfw": _settings.threshold_nsfw,
            "nsfl": _settings.threshold_nsfl,
            "suggestive": _settings.threshold_suggestive,
        },
    )
    return success_response(request, result.model_dump(mode="json"))


# ---------------------------------------------------------------------------
#  Yordamchilar
# ---------------------------------------------------------------------------


async def _json_body(request: Request) -> dict:
    raw = await request.body()
    if not raw:
        raise ApiError(
            ErrorCode.INVALID_REQUEST, "So'rov tanasi bo'sh", status_code=400
        )
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(
            ErrorCode.INVALID_REQUEST, "So'rov tanasi yaroqli JSON emas", status_code=400
        ) from exc
    if not isinstance(body, dict):
        raise ApiError(
            ErrorCode.INVALID_REQUEST, "So'rov tanasi JSON obyekt bo'lishi kerak", status_code=400
        )
    return body


def _parse_options(raw: str | None) -> AnalyzeRequest:
    """multipart rejimidagi ixtiyoriy `options` JSON matnini o'qiydi."""
    from app.schemas import AnalyzeOptions

    if not raw:
        return AnalyzeOptions()  # type: ignore[return-value]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(
            ErrorCode.INVALID_REQUEST, "`options` yaroqli JSON emas", status_code=400
        ) from exc
    try:
        return AnalyzeOptions.model_validate(parsed)  # type: ignore[return-value]
    except ValueError as exc:
        raise ApiError(ErrorCode.INVALID_REQUEST, _first_error(exc), status_code=400) from exc


def _first_error(exc: ValueError) -> str:
    """Pydantic xatosidan mijozga tushunarli bitta xabar ajratib oladi."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            first = errors()[0]
            location = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
            message = first.get("msg", "noto'g'ri qiymat")
            return f"{location}: {message}" if location else message
        except (IndexError, KeyError, TypeError):
            pass
    return str(exc).split("\n")[0]


__all__ = ["router", "error_body"]
