"""Tahlil quvuri: baytlar → dekod → klassifikator → detektor → verdict."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path

from pydantic import ValidationError

from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.core import fetcher, imaging, loader
from app.envelope import ApiError, ErrorCode
from app.schemas import AnalyzeResult, BatchItem, Detection, Timings
from app.services import reasons as reason_texts
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
        #: Kesh kalitiga qo'shiladigan versiya — `compute_fingerprint()`.
        self.fingerprint = "nofp"

    @property
    def ready(self) -> bool:
        return self.classifier is not None


engine = Engine()


def _file_digest(path: Path | None) -> str:
    if path is None or not path.exists():
        return "-"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_fingerprint(
    classifier_path: Path, head_path: Path | None, detector_path: Path | None
) -> str:
    """Natijaga ta'sir qiladigan HAMMA narsaning qisqa dayjesti.

    Kesh kaliti ilgari faqat `rasm_sha256:detect:min_score` edi — chegaralarni
    qayta sozlash, o'z boshini almashtirish yoki NudeNet modelini yangilashdan
    keyin eski verdictlar yana 7 kun (CACHE_TTL) keshdan qaytardi. Endi
    fingerprint o'zgarsa eski yozuvlar shunchaki o'qilmaydi va TTL bilan o'ladi.
    """
    parts = [
        _settings.version,
        _file_digest(classifier_path),
        _file_digest(head_path),
        _file_digest(detector_path),
        f"{_settings.threshold_nsfw}/{_settings.threshold_nsfl}/"
        f"{_settings.threshold_suggestive}/{_settings.threshold_nsfw_confident}/"
        f"{_settings.detection_min_score}",
        f"{scoring.EXPLICIT_PROMOTE_SCORE}/{scoring.SUGGESTIVE_PROMOTE_SCORE}/"
        f"{scoring.REVIEW_MARGIN}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _rescale(raw: dict, scale: float) -> dict:
    """Topilma ramkasini ishchi rasmdan asl koordinatalarga qaytaradi."""
    if abs(scale - 1.0) < 1e-9:
        return raw
    box = dict(raw["box"])
    for key in ("x", "y", "width", "height"):
        box[key] = int(round(box[key] * scale))
    return {**raw, "box": box}


async def _resolve_bytes(item: BatchItem, timings: dict[str, int]) -> bytes:
    """Kirish rejimiga qarab rasm baytlarini oladi."""
    started = time.perf_counter()
    if item.url:
        data = await fetcher.fetch_image(item.url)
    elif item.path:
        data = await run_in_threadpool(loader.load_from_path, item.path)
    elif item.image_base64:
        data = await run_in_threadpool(loader.load_from_base64, item.image_base64)
    else:  # schemas.py validatori buni oldini oladi — himoya uchun
        raise ApiError(
            ErrorCode.INVALID_REQUEST, "Kirish manbasi berilmagan", status_code=400
        )
    timings["fetch"] = int((time.perf_counter() - started) * 1000)
    return data


def _infer(
    data: bytes, detect: bool, min_score: float, sha256: str | None = None
) -> tuple[AnalyzeResult, dict]:
    """Sinxron og'ir qism — threadpool'da chaqiriladi (event loop bloklanmasin)."""
    timings: dict[str, int] = {}

    t0 = time.perf_counter()
    img, info, scale = imaging.decode(data, sha256=sha256)
    timings["decode"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    class_scores = engine.classifier.predict(img)
    timings["classify"] = int((time.perf_counter() - t0) * 1000)

    # MUHIM (2026-09-25): verdict HAR DOIM serverning qat'iy chegarasi
    # (`DETECTION_MIN_SCORE`) bo'yicha filtrlangan topilmalardan hisoblanadi.
    # Mijozning `min_detection_score` i FAQAT javobdagi ro'yxatni filtrlaydi.
    # Ilgari u detektorning o'ziga uzatilardi: bir xil rasm 25 da `nsfw`,
    # 70 da `suggestive`/`is_nsfw=false` chiqardi — ya'ni mijoz (UI'dagi
    # "shovqin"ni kamaytirmoqchi bo'lib) moderatsiyani bilmasdan o'chirardi.
    floor = _settings.detection_min_score
    detections: list[Detection] = []
    scoring_detections: list[Detection] = []
    timings["detect"] = 0
    detector_used = detect and engine.detector is not None
    if detector_used:
        t0 = time.perf_counter()
        # Detektorsiz verdict PASTGA og'adi: klassifikator 15% bergan
        # ochiq rasm `safe` bo'lib chiqib ketardi (tests/test_scoring.py:37).
        # Shuning uchun mijozga "safe" deyish o'rniga 503 qaytariladi —
        # EGA QARORI 2026-09-28. Bir marta qayta urinamiz: ONNX ning
        # o'tkinchi nosozligi butun so'rovni yo'qotmasin.
        for attempt in (1, 2):
            try:
                bgr = imaging.to_bgr_array(img)
                raw = engine.detector.detect(bgr, min(floor, min_score))
                all_detections = [Detection(**_rescale(d, scale)) for d in raw]
                scoring_detections = [d for d in all_detections if d.score >= floor]
                detections = [d for d in all_detections if d.score >= min_score]
                break
            except Exception:  # noqa: BLE001
                log.exception("detektor ishlamadi (urinish %s)", attempt)
                if attempt == 2:
                    raise ApiError(
                        ErrorCode.SERVICE_UNAVAILABLE,
                        "Tana qismlari detektori vaqtincha ishlamayapti",
                        status_code=503,
                    ) from None
        timings["detect"] = int((time.perf_counter() - t0) * 1000)

    verdict, confidence, scores, reasons = scoring.evaluate(
        class_scores, scoring_detections, detections_available=detector_used
    )

    review, review_reason = scoring.needs_review(verdict, class_scores, scoring_detections)
    if review_reason is not None:
        reasons.append(review_reason)

    # O'lchangan: 1600x1157 rasmda `BUTTOCKS_EXPOSED 37%` va
    # `ARMPITS_EXPOSED 47%` topilgan, o'sha rasmning 256x185 thumbnail'ida
    # ikkalasi ham yo'qolgan. NudeNet kirishi 320x320 — undan kichik rasm
    # kattalashtiriladi va topilmalar zaiflashadi. Bu verdictni
    # o'zgartirmaydi, faqat mijozga "asl rasmni yubor" deb aytadi.
    if detector_used and min(info.width, info.height) < _DETECTOR_INPUT_PX:
        reasons.append(
            ("small_image",
             {"width": info.width, "height": info.height,
              "input": _DETECTOR_INPUT_PX})
        )

    result = AnalyzeResult(
        verdict=verdict,
        is_nsfw=scoring.is_nsfw(verdict),
        is_safe=not scoring.is_nsfw(verdict),
        needs_review=review,
        confidence=confidence,
        scores=scores,
        detections=detections,
        reasons=[reason_texts.render(code, params) for code, params in reasons],
        image=info,
        models={
            "classifier": CLASSIFIER_NAME,
            "detector": DETECTOR_NAME if detector_used else "disabled",
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
    # `min_score` endi faqat ko'rsatiladigan ro'yxatga ta'sir qiladi, lekin
    # javob tanasi baribir farq qiladi — shuning uchun kalitda qoladi.
    cache_key = f"{engine.fingerprint}:{sha}:{int(detect)}:{min_score:g}"

    if use_cache:
        cached = await cache.get_result(cache_key)
        if cached:
            try:
                result = AnalyzeResult.model_validate(cached)
            except ValidationError:
                # Eskirgan shakldagi yozuv — bu SERVER muammosi; ilgari u
                # `400 INVALID_REQUEST` bo'lib mijozga yozilardi.
                log.warning("keshdagi yozuv shaklga mos emas — qayta hisoblanadi")
                result = None
        else:
            result = None
        if result is not None:
            result.cached = True
            result.timings_ms = Timings(
                fetch=fetch_ms,
                total=fetch_ms + int((time.perf_counter() - total_started) * 1000),
            )
            return result

    result, _timings = await run_in_threadpool(_infer, data, detect, min_score, sha)
    result.timings_ms.fetch = fetch_ms
    result.timings_ms.total = fetch_ms + int((time.perf_counter() - total_started) * 1000)

    if use_cache:
        payload = result.model_dump(mode="json")
        payload["cached"] = False
        await cache.set_result(cache_key, payload)

    return result


async def analyze_item(item: BatchItem) -> AnalyzeResult:
    try:
        async with asyncio.timeout(_settings.item_total_timeout):
            return await _analyze_item(item)
    except TimeoutError as exc:
        raise ApiError(
            ErrorCode.FETCH_TIMEOUT,
            "Tahlil vaqti tugadi",
            status_code=504,
            details={"timeout_s": _settings.item_total_timeout},
        ) from exc


async def _analyze_item(item: BatchItem) -> AnalyzeResult:
    timings: dict[str, int] = {"fetch": 0}
    data = await _resolve_bytes(item, timings)
    return await analyze_bytes(
        data,
        detect=item.detect,
        min_score=item.min_detection_score,
        use_cache=item.cache,
        fetch_ms=timings["fetch"],
    )
