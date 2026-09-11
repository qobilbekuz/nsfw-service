"""Tahlil quvuri: baytlar → dekod → klassifikator → detektor → verdict."""

from __future__ import annotations

import hashlib
import logging
import time

from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.core import fetcher, imaging, loader
from app.envelope import ApiError, ErrorCode
from app.schemas import AnalyzeResult, BatchItem, Detection, Timings
from app.services import scoring
from app.services.cache import cache
from app.services.classifier import MODEL_NAME as CLASSIFIER_NAME
from app.services.detector import MODEL_NAME as DETECTOR_NAME

_settings = get_settings()
log = logging.getLogger("nsfw.pipeline")

#: NudeNet kirish o'lchami. Rasmning kichik tomoni shundan past bo'lsa,
#: detektor uni kattalashtiradi va topilmalar zaiflashadi (izohga qarang).
_DETECTOR_INPUT_PX = 320


class Engine:
    """Yuklangan modellarni ushlab turadi (lifespan davomida bitta nusxa)."""

    def __init__(self) -> None:
        self.classifier = None
        self.detector = None

    @property
    def ready(self) -> bool:
        return self.classifier is not None


engine = Engine()


async def _resolve_bytes(item: BatchItem, timings: dict[str, int]) -> bytes:
    """Kirish rejimiga qarab rasm baytlarini oladi."""
    started = time.perf_counter()
    if item.url:
        data = await fetcher.fetch_image(item.url)
    elif item.path:
        data = await run_in_threadpool(loader.load_from_path, item.path)
    elif item.image_base64:
        data = loader.load_from_base64(item.image_base64)
    else:  # schemas.py validatori buni oldini oladi — himoya uchun
        raise ApiError(
            ErrorCode.INVALID_REQUEST, "Kirish manbasi berilmagan", status_code=400
        )
    timings["fetch"] = int((time.perf_counter() - started) * 1000)
    return data


def _infer(data: bytes, detect: bool, min_score: float) -> tuple[AnalyzeResult, dict]:
    """Sinxron og'ir qism — threadpool'da chaqiriladi (event loop bloklanmasin)."""
    timings: dict[str, int] = {}

    t0 = time.perf_counter()
    img, info = imaging.decode(data)
    timings["decode"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    class_scores = engine.classifier.predict(img)
    timings["classify"] = int((time.perf_counter() - t0) * 1000)

    detections: list[Detection] = []
    timings["detect"] = 0
    detector_used = detect and engine.detector is not None
    detector_failed = False
    if detector_used:
        t0 = time.perf_counter()
        try:
            bgr = imaging.to_bgr_array(img)
            raw = engine.detector.detect(bgr, min_score)
            detections = [Detection(**d) for d in raw]
        except Exception:  # noqa: BLE001
            # Klassifikator allaqachon javob berdi — detektor yiqilgani uchun
            # butun so'rovni 500 bilan yo'qotish isrof. Detektorsiz davom
            # etamiz: oraliq zonada bu qaror `nsfw` tomonga og'adi (xavfsiz
            # yo'nalish), natija esa har doim `needs_review` bo'ladi.
            log.exception("detektor ishlamadi — klassifikator bilan davom etiladi")
            detector_used = False
            detector_failed = True
            detections = []
        timings["detect"] = int((time.perf_counter() - t0) * 1000)

    verdict, confidence, scores, reasons = scoring.evaluate(
        class_scores, detections, detections_available=detector_used
    )

    review, review_reason = scoring.needs_review(verdict, class_scores, detections)
    if review_reason:
        reasons.append(review_reason)

    if detector_failed:
        reasons.append("diqqat: detektor ishlamadi, qaror faqat klassifikatorga tayandi")
        review = True

    # O'lchangan: 1600x1157 rasmda `BUTTOCKS_EXPOSED 37%` va
    # `ARMPITS_EXPOSED 47%` topilgan, o'sha rasmning 256x185 thumbnail'ida
    # ikkalasi ham yo'qolgan. NudeNet kirishi 320x320 — undan kichik rasm
    # kattalashtiriladi va topilmalar zaiflashadi. Bu verdictni
    # o'zgartirmaydi, faqat mijozga "asl rasmni yubor" deb aytadi.
    if detector_used and min(info.width, info.height) < _DETECTOR_INPUT_PX:
        reasons.append(
            f"diqqat: rasm kichik ({info.width}x{info.height} < "
            f"{_DETECTOR_INPUT_PX}px) — detektor topilmalari zaiflashadi, "
            "iloji bo'lsa asl rasmni yuboring"
        )

    result = AnalyzeResult(
        verdict=verdict,
        is_nsfw=scoring.is_nsfw(verdict),
        is_safe=not scoring.is_nsfw(verdict),
        needs_review=review,
        confidence=confidence,
        scores=scores,
        detections=detections,
        reasons=reasons,
        image=info,
        models={
            "classifier": CLASSIFIER_NAME,
            "detector": (
                DETECTOR_NAME
                if detector_used
                else ("failed" if detector_failed else "disabled")
            ),
        },
        timings_ms=Timings(**timings),
        cached=False,
    )
    return result, timings


async def analyze_bytes(
    data: bytes,
    *,
    detect: bool = True,
    min_score: float = 25.0,
    use_cache: bool = True,
    fetch_ms: int = 0,
) -> AnalyzeResult:
    if not engine.ready:
        raise ApiError(
            ErrorCode.SERVICE_UNAVAILABLE,
            "Modellar hali yuklanmagan",
            status_code=503,
        )

    total_started = time.perf_counter()
    imaging.check_size(len(data))

    # Kesh kaliti: rasm sha256 + tahlil parametrlari. Parametrlarni kalitga
    # qo'shmasak, `detect=false` bilan olingan natija `detect=true` so'roviga
    # ham qaytarilib, topilmalar tushib qolardi.
    sha = hashlib.sha256(data).hexdigest()
    cache_key = f"{sha}:{int(detect)}:{min_score:g}"

    if use_cache:
        cached = await cache.get_result(cache_key)
        if cached:
            result = AnalyzeResult.model_validate(cached)
            result.cached = True
            result.timings_ms = Timings(
                fetch=fetch_ms,
                total=fetch_ms + int((time.perf_counter() - total_started) * 1000),
            )
            return result

    result, _timings = await run_in_threadpool(_infer, data, detect, min_score)
    result.timings_ms.fetch = fetch_ms
    result.timings_ms.total = fetch_ms + int((time.perf_counter() - total_started) * 1000)

    if use_cache:
        payload = result.model_dump(mode="json")
        payload["cached"] = False
        await cache.set_result(cache_key, payload)

    return result


async def analyze_item(item: BatchItem) -> AnalyzeResult:
    timings: dict[str, int] = {"fetch": 0}
    data = await _resolve_bytes(item, timings)
    return await analyze_bytes(
        data,
        detect=item.detect,
        min_score=item.min_detection_score,
        use_cache=item.cache,
        fetch_ms=timings["fetch"],
    )
