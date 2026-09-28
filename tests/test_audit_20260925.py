"""2026-09-25 auditida topilgan nuqsonlarning regressiya testlari.

Har bir test aynan o'lchangan nuqsonni qayta tiklashga urinadi — tuzatish
olib tashlansa, test yiqilishi kerak.
"""

from __future__ import annotations

import asyncio
import base64
import time

import pytest

from app.api import deps
from app.config import get_settings
from app.services import cache as cache_mod
from app.services import pipeline
from app.services.cache import FOUND, MISSING, UNAVAILABLE, CacheClient


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# --------------------------------------------------------------------------
#  1. min_detection_score verdictni o'zgartirmasligi kerak
# --------------------------------------------------------------------------


def _fake_models(monkeypatch, nsfw: float, detections: list[dict]) -> None:
    """Klassifikator oraliq zonada (60-90), detektor bitta topilma beradi."""
    monkeypatch.setattr(
        pipeline.engine.classifier,
        "predict",
        lambda _img: {"sfw": 100.0 - nsfw - 5.0, "nsfw": nsfw, "nsfl": 5.0},
    )

    def fake_detect(_bgr, min_score):
        # Haqiqiy detektor kabi: `min_score` dan past topilmalar tashlanadi.
        return [d for d in detections if d["score"] >= min_score]

    monkeypatch.setattr(pipeline.engine.detector, "detect", fake_detect)


BREAST_62 = {
    "label": "FEMALE_BREAST_EXPOSED",
    "score": 62.0,
    "box": {"x": 0, "y": 0, "width": 10, "height": 10},
}


@pytest.mark.parametrize("min_score", [0, 25, 70, 100])
def test_min_detection_score_does_not_change_verdict(client, image_bytes, monkeypatch, min_score):
    """Audit: ilgari 25 da `nsfw`, 70 da `suggestive`/`is_nsfw=false` edi."""
    _fake_models(monkeypatch, nsfw=75.0, detections=[BREAST_62])

    resp = client.post(
        "/v1/analyze",
        json={"image_base64": _b64(image_bytes), "min_detection_score": min_score},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    assert data["verdict"] == "nsfw"
    assert data["is_nsfw"] is True
    # Ko'rsatiladigan ro'yxat esa mijoz chegarasiga bo'ysunadi.
    shown = [d["label"] for d in data["detections"]]
    assert shown == (["FEMALE_BREAST_EXPOSED"] if min_score <= 62 else [])


def test_low_client_threshold_shows_more_but_scores_on_server_floor(client, image_bytes, monkeypatch):
    """Mijoz 10% so'rasa ko'proq topilma ko'radi, lekin 25% dan pastlari
    verdictga (va needs_review ga) ta'sir qilmaydi."""
    weak = {"label": "FEMALE_BREAST_EXPOSED", "score": 12.0, "box": BREAST_62["box"]}
    _fake_models(monkeypatch, nsfw=10.0, detections=[weak])

    data = client.post(
        "/v1/analyze",
        json={"image_base64": _b64(image_bytes), "min_detection_score": 10},
    ).json()["data"]

    assert [d["score"] for d in data["detections"]] == [12.0]
    assert data["verdict"] == "safe"
    assert data["needs_review"] is False


# --------------------------------------------------------------------------
#  2. GET da noto'g'ri parametr 500 emas, 400
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["500", "-1", "abc"])
def test_get_invalid_min_detection_score_is_400(client, value):
    resp = client.get(
        "/v1/analyze", params={"url": "https://example.com/a.jpg", "min_detection_score": value}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_get_accepts_both_cache_names(client, monkeypatch):
    seen = []

    async def fake_analyze_item(item):
        seen.append(item.cache)
        raise pipeline.ApiError("FETCH_FAILED", "x", status_code=502)

    monkeypatch.setattr(pipeline, "analyze_item", fake_analyze_item)
    for params in ({"cache": "false"}, {"cache_enabled": "false"}, {}):
        client.get("/v1/analyze", params={"url": "https://example.com/a.jpg", **params})
    assert seen == [False, False, True]


# --------------------------------------------------------------------------
#  3. Batch rasm soniga teng limit birligi yeydi
# --------------------------------------------------------------------------


def test_batch_charges_one_unit_per_item(client, image_bytes, monkeypatch):
    costs: list[int] = []

    async def fake_enforce(_request, _principal, cost=1):
        costs.append(cost)
        return {}

    import app.api.routes as routes

    monkeypatch.setattr(routes, "enforce_rate_limit", fake_enforce)
    items = [{"image_base64": _b64(image_bytes), "id": str(i)} for i in range(7)]
    resp = client.post("/v1/analyze/batch", json={"items": items})
    assert resp.status_code == 200, resp.text
    assert costs == [7]


def test_rate_limit_lua_uses_cost():
    assert "INCRBY" in cache_mod._RATE_LUA
    assert "ARGV[2]" in cache_mod._RATE_LUA


# --------------------------------------------------------------------------
#  4. Kesh kalitida model/chegara versiyasi (fingerprint)
# --------------------------------------------------------------------------


def test_fingerprint_changes_with_thresholds(monkeypatch, tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"model-v1")
    settings = get_settings()

    base = pipeline.compute_fingerprint(model, None, None)
    assert base == pipeline.compute_fingerprint(model, None, None)

    monkeypatch.setattr(settings, "threshold_nsfw", settings.threshold_nsfw + 1)
    assert pipeline.compute_fingerprint(model, None, None) != base
    monkeypatch.undo()

    model.write_bytes(b"model-v2")
    assert pipeline.compute_fingerprint(model, None, None) != base


def test_cache_key_contains_fingerprint(client, image_bytes, monkeypatch):
    keys: list[str] = []

    async def fake_get(key):
        keys.append(key)
        return None

    monkeypatch.setattr(pipeline.cache, "get_result", fake_get)
    monkeypatch.setattr(pipeline.engine, "fingerprint", "fp_test_123")
    client.post("/v1/analyze", json={"image_base64": _b64(image_bytes)})
    assert keys and keys[0].startswith("fp_test_123:")


# --------------------------------------------------------------------------
#  5. Zaxira kalitlar (Redis o'chsa ham auth ishlaydi, lekin fail-closed)
# --------------------------------------------------------------------------


class _Req:
    def __init__(self, key: str) -> None:
        self.headers = {"x-api-key": key}
        self.client = type("C", (), {"host": "203.0.113.9"})()


@pytest.mark.parametrize("state", [UNAVAILABLE, MISSING])
def test_bootstrap_key_works_without_redis(monkeypatch, state):
    key = "nsfw_test_bootstrap"
    digest = CacheClient.hash_key(key)

    async def fake_lookup(_k):
        return state, None

    monkeypatch.setattr(deps.cache, "lookup_key", fake_lookup)
    monkeypatch.setattr(deps._settings, "bootstrap_keys", [f"edu:{digest}:90"])
    monkeypatch.setattr(deps._settings, "public_mode", False)

    principal = asyncio.run(deps.authenticate(_Req(key)))
    assert principal.kind == "key" and principal.id == "edu" and principal.rate_limit == 90


def test_unknown_key_still_rejected_without_redis(monkeypatch):
    async def fake_lookup(_k):
        return UNAVAILABLE, None

    monkeypatch.setattr(deps.cache, "lookup_key", fake_lookup)
    monkeypatch.setattr(deps._settings, "bootstrap_keys", [f"edu:{'a' * 64}"])
    monkeypatch.setattr(deps._settings, "public_mode", True)  # public ham kalitni oqlamaydi

    with pytest.raises(deps.ApiError) as exc:
        asyncio.run(deps.authenticate(_Req("nsfw_wrong")))
    assert exc.value.status_code == 401


def test_revoked_redis_key_wins_over_nothing(monkeypatch):
    async def fake_lookup(_k):
        return FOUND, {"id": "k1", "revoked": True}

    monkeypatch.setattr(deps.cache, "lookup_key", fake_lookup)
    with pytest.raises(deps.ApiError):
        asyncio.run(deps.authenticate(_Req("nsfw_revoked")))


# --------------------------------------------------------------------------
#  6. Redis'ga qayta ulanish
# --------------------------------------------------------------------------


def test_cache_client_retries_connection(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0")  # yopiq port
    monkeypatch.setattr(settings, "redis_retry_seconds", 0.2)

    client = CacheClient()
    attempts = []
    real_connect = client.connect

    async def counting_connect():
        attempts.append(time.monotonic())
        await real_connect()

    monkeypatch.setattr(client, "connect", counting_connect)

    async def scenario():
        assert await client.lookup_key("x") == (UNAVAILABLE, None)  # 1-urinish
        assert await client.lookup_key("x") == (UNAVAILABLE, None)  # oraliq ichida — urinmaydi
        assert len(attempts) == 1
        await asyncio.sleep(0.25)
        await client.lookup_key("x")  # oraliq o'tdi — yana urinadi
        assert len(attempts) == 2

    asyncio.run(scenario())


# --------------------------------------------------------------------------
#  7. Envelope va X-Request-ID
# --------------------------------------------------------------------------


def test_envelope_has_standard_fields(client):
    ok = client.get("/v1/health").json()
    assert ok["success"] is True and ok["ok"] is True and ok["status_code"] == 200

    err = client.get("/v1/does-not-exist").json()
    assert err["ok"] is False and err["status_code"] == 404
    assert set(err["error"]["messages"]) == {"uz", "ru", "en"}


def test_request_id_is_validated(client):
    good = client.get("/v1/health", headers={"X-Request-ID": "abc-123"})
    assert good.headers["x-request-id"] == "abc-123"

    bad = client.get("/v1/health", headers={"X-Request-ID": "A" * 3000})
    assert bad.headers["x-request-id"].startswith("r_")
    assert len(bad.headers["x-request-id"]) < 64
