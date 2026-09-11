"""Detektor yiqilganda va rasm juda kichik bo'lganda quvur nima qiladi.

Ikkalasi ham verdictni "to'g'rilamaydi" — maqsad shundaki, ishonchsiz holat
javobda ko'rinsin va jimgina yaxshi natijaday ko'rinmasin.
"""

from __future__ import annotations

from app.services import pipeline
from tests.conftest import make_image


def _analyze(client, data: bytes) -> dict:
    import base64

    resp = client.post(
        "/v1/analyze",
        json={"image_base64": base64.b64encode(data).decode()},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]


def test_detector_failure_degrades_instead_of_500(client, image_bytes, monkeypatch):
    """Detektor xato bersa ham javob qaytadi — faqat belgilangan holda."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("onnxruntime crashed")

    monkeypatch.setattr(pipeline.engine.detector, "detect", boom)

    data = _analyze(client, image_bytes)

    assert data["models"]["detector"] == "failed"
    assert data["detections"] == []
    # Detektorsiz chiqqan qaror har doim qo'lda ko'rib chiqishga tushadi.
    assert data["needs_review"] is True
    assert any("detektor ishlamadi" in r for r in data["reasons"])


def test_detector_failure_does_not_leak_into_cache_semantics(client, image_bytes):
    """Oldingi test monkeypatch'ni qaytargach, detektor yana ishlashi kerak."""
    data = _analyze(client, image_bytes)
    assert data["models"]["detector"] == "nudenet-320n"


def test_small_image_is_flagged_in_reasons(client):
    """256x185 thumbnail — NudeNet kirishidan (320) kichik."""
    data = _analyze(client, make_image(size=(256, 185)))
    assert any("rasm kichik" in r for r in data["reasons"])


def test_large_enough_image_has_no_size_warning(client):
    data = _analyze(client, make_image(size=(640, 480)))
    assert not any("rasm kichik" in r for r in data["reasons"])


def test_size_warning_does_not_change_verdict(client):
    """Ogohlantirish faqat matn — verdict va bayroqlarga tegmaydi."""
    small = _analyze(client, make_image(size=(256, 185), color=(200, 170, 150)))
    big = _analyze(client, make_image(size=(640, 480), color=(200, 170, 150)))
    assert small["verdict"] == big["verdict"] == "safe"
    assert small["is_safe"] is big["is_safe"] is True
