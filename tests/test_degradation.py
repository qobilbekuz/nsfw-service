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


def test_detector_failure_is_503_not_a_safe_verdict(client, image_bytes, monkeypatch):
    """EGA QARORI 2026-09-28: detektorsiz "xavfsiz" deyilmaydi.

    Ilgari javob klassifikatorga tayanib qaytardi — ya'ni ochiq rasm
    `is_safe: true` bo'lib chiqib ketishi mumkin edi
    (`tests/test_scoring.py:37` aynan shu holatni ko'rsatadi).
    """
    import base64

    def boom(*_args, **_kwargs):
        raise RuntimeError("onnxruntime crashed")

    monkeypatch.setattr(pipeline.engine.detector, "detect", boom)

    resp = client.post(
        "/v1/analyze",
        json={"image_base64": base64.b64encode(image_bytes).decode()},
    )
    assert resp.status_code == 503, resp.text
    assert resp.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


def test_detector_failure_does_not_leak_into_cache_semantics(client, image_bytes):
    """Oldingi test monkeypatch'ni qaytargach, detektor yana ishlashi kerak."""
    data = _analyze(client, image_bytes)
    assert data["models"]["detector"] == "nudenet-320n"


def test_small_image_is_flagged_in_reasons(client):
    """256x185 thumbnail — NudeNet kirishidan (320) kichik."""
    data = _analyze(client, make_image(size=(256, 185)))
    assert any(r["code"] == "small_image" for r in data["reasons"])


def test_reason_matni_UCH_TILDA(client):
    """Xizmat uch tilli platformaga xizmat qiladi — sabab ham uch tilda."""
    data = _analyze(client, make_image(size=(640, 480)))
    first = data["reasons"][0]
    assert first["code"] == "classifier_scores"
    assert set(first["messages"]) == {"en", "uz", "ru"}
    assert all(text.strip() for text in first["messages"].values())
    assert first["params"]["sfw"] > 0


def test_large_enough_image_has_no_size_warning(client):
    data = _analyze(client, make_image(size=(640, 480)))
    assert not any(r["code"] == "small_image" for r in data["reasons"])


def test_size_warning_does_not_change_verdict(client):
    """Ogohlantirish faqat matn — verdict va bayroqlarga tegmaydi."""
    small = _analyze(client, make_image(size=(256, 185), color=(200, 170, 150)))
    big = _analyze(client, make_image(size=(640, 480), color=(200, 170, 150)))
    assert small["verdict"] == big["verdict"] == "safe"
    assert small["is_safe"] is big["is_safe"] is True
