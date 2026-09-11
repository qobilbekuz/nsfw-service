"""API kontrakti: envelope, kirish rejimlari, batch, xizmat endpointlari."""

from __future__ import annotations

import base64

from tests.conftest import make_image

ENVELOPE_KEYS = {"success", "request_id", "took_ms", "data", "error"}


def assert_envelope(payload: dict) -> None:
    assert set(payload.keys()) == ENVELOPE_KEYS
    assert isinstance(payload["request_id"], str) and payload["request_id"]
    assert isinstance(payload["took_ms"], int)
    # `success` va `error` hech qachon birga to'ldirilmaydi.
    if payload["success"]:
        assert payload["error"] is None and payload["data"] is not None
    else:
        assert payload["data"] is None and payload["error"] is not None
        assert set(payload["error"].keys()) == {"code", "message", "details"}


# --------------------------------------------------------------------------
#  Envelope barcha yo'llarda bir xilmi
# --------------------------------------------------------------------------


def test_envelope_on_success(client, image_bytes) -> None:
    response = client.post("/v1/analyze", files={"file": ("a.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 200
    assert_envelope(response.json())


def test_envelope_on_error(client) -> None:
    response = client.post("/v1/analyze", json={})
    assert response.status_code == 400
    assert_envelope(response.json())


def test_envelope_on_404(client) -> None:
    response = client.get("/v1/mavjud-emas")
    assert response.status_code == 404
    assert_envelope(response.json())
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_request_id_echoed(client) -> None:
    response = client.get("/v1/health", headers={"X-Request-ID": "r_mijoz_bergan"})
    assert response.headers["x-request-id"] == "r_mijoz_bergan"
    assert response.json()["request_id"] == "r_mijoz_bergan"


# --------------------------------------------------------------------------
#  Kirish rejimlari
# --------------------------------------------------------------------------


def test_multipart_mode(client, image_bytes) -> None:
    response = client.post("/v1/analyze", files={"file": ("a.jpg", image_bytes, "image/jpeg")})
    data = response.json()["data"]
    assert data["verdict"] in {"safe", "suggestive", "nsfw", "nsfl"}
    assert data["image"]["width"] == 256
    assert len(data["image"]["sha256"]) == 64


def test_base64_mode(client, image_bytes) -> None:
    payload = base64.b64encode(image_bytes).decode()
    response = client.post("/v1/analyze", json={"image_base64": payload})
    assert response.status_code == 200
    assert response.json()["data"]["image"]["format"] == "JPEG"


def test_base64_data_uri_prefix(client, image_bytes) -> None:
    payload = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
    response = client.post("/v1/analyze", json={"image_base64": payload})
    assert response.status_code == 200


def test_exactly_one_source_required(client) -> None:
    for body in ({}, {"url": "https://a.co/x.jpg", "path": "y.jpg"}):
        response = client.post("/v1/analyze", json=body)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_unknown_field_rejected(client) -> None:
    response = client.post("/v1/analyze", json={"url": "https://a.co/x.jpg", "evil": 1})
    assert response.status_code == 400


def test_malformed_json_rejected(client) -> None:
    response = client.post(
        "/v1/analyze", content=b"{oops", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------
#  Ballar shakli
# --------------------------------------------------------------------------


def test_scores_are_percentages(client, image_bytes) -> None:
    data = client.post(
        "/v1/analyze", files={"file": ("a.jpg", image_bytes, "image/jpeg")}
    ).json()["data"]
    scores = data["scores"]
    for value in scores.values():
        assert 0.0 <= value <= 100.0
    # sfw + nsfw + nsfl — klassifikator softmaxi, yig'indisi 100 ga teng.
    total = scores["sfw"] + scores["nsfw"] + scores["nsfl"]
    assert abs(total - 100.0) < 0.5
    assert data["is_nsfw"] is not data["is_safe"]


def test_detect_flag_disables_detector(client, image_bytes) -> None:
    payload = base64.b64encode(image_bytes).decode()
    response = client.post("/v1/analyze", json={"image_base64": payload, "detect": False})
    data = response.json()["data"]
    assert data["detections"] == []
    assert data["models"]["detector"] == "disabled"


# --------------------------------------------------------------------------
#  Batch
# --------------------------------------------------------------------------


def test_batch_mixed_results(client, image_bytes) -> None:
    payload = base64.b64encode(image_bytes).decode()
    response = client.post(
        "/v1/analyze/batch",
        json={
            "items": [
                {"id": "ok", "image_base64": payload},
                {"id": "ssrf", "url": "http://127.0.0.1:8888/"},
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["count"] == 2 and data["succeeded"] == 1 and data["failed"] == 1

    by_id = {item["id"]: item for item in data["results"]}
    assert by_id["ok"]["success"] is True
    assert by_id["ssrf"]["success"] is False
    # Bitta element xato bo'lsa ham butun batch 200 qaytadi.
    assert by_id["ssrf"]["error"]["code"] == "FORBIDDEN_TARGET"


def test_batch_limit_enforced(client, image_bytes) -> None:
    payload = base64.b64encode(image_bytes).decode()
    response = client.post(
        "/v1/analyze/batch", json={"items": [{"image_base64": payload}] * 25}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_batch_empty_rejected(client) -> None:
    assert client.post("/v1/analyze/batch", json={"items": []}).status_code == 400


# --------------------------------------------------------------------------
#  Xizmat endpointlari
# --------------------------------------------------------------------------


def test_health_needs_no_auth(client) -> None:
    data = client.get("/v1/health").json()["data"]
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


def test_models_endpoint(client) -> None:
    data = client.get("/v1/models").json()["data"]
    assert data["classifier"]["name"] == "image-safety-classifier-s"
    assert data["detector"]["classes"] == 18
    assert "JPEG" in data["limits"]["allowed_formats"]


def test_formats_supported(client) -> None:
    for fmt, mime in (("PNG", "image/png"), ("WEBP", "image/webp"), ("BMP", "image/bmp")):
        data = make_image(fmt=fmt)
        response = client.post("/v1/analyze", files={"file": (f"a.{fmt}", data, mime)})
        assert response.status_code == 200, fmt
        assert response.json()["data"]["image"]["format"] == fmt
