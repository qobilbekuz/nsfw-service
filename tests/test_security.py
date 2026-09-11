"""SSRF, path traversal va kirish limitlari testlari (TASKS.md §8)."""

from __future__ import annotations

import base64

import pytest

from app.core import netguard

# --------------------------------------------------------------------------
#  netguard birlik testlari
# --------------------------------------------------------------------------

INTERNAL_IPS = [
    "127.0.0.1",
    "0.0.0.0",
    "10.0.0.1",
    "172.16.0.1",
    "192.168.1.1",
    "169.254.169.254",  # bulut metadata endpointi
    "100.64.0.1",  # CGNAT
    "224.0.0.1",  # multicast
    "::1",
    "fe80::1",
    "fc00::1",
    "::ffff:127.0.0.1",  # IPv4-mapped loopback
    "::ffff:10.0.0.1",
    "2001:db8::1",
]

PUBLIC_IPS = ["8.8.8.8", "1.1.1.1", "144.76.201.78", "2a00:1450:4001:80f::200e"]


@pytest.mark.parametrize("ip", INTERNAL_IPS)
def test_internal_ips_rejected(ip: str) -> None:
    assert netguard.is_public(ip) is False


@pytest.mark.parametrize("ip", PUBLIC_IPS)
def test_public_ips_allowed(ip: str) -> None:
    assert netguard.is_public(ip) is True


def test_mixed_dns_result_rejected() -> None:
    """Bitta tashqi + bitta ichki IP — bu DNS rebinding naqshi."""
    import socket

    mixed = [(socket.AF_INET, "8.8.8.8"), (socket.AF_INET, "127.0.0.1")]
    assert netguard.pick_safe_ip(mixed) is None
    assert netguard.pick_safe_ip([(socket.AF_INET, "8.8.8.8")]) is not None


# --------------------------------------------------------------------------
#  API darajasidagi SSRF
# --------------------------------------------------------------------------

SSRF_URLS = [
    "http://127.0.0.1:8888/",
    "http://localhost:6379/",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/a.jpg",
    "http://192.168.1.1/a.jpg",
    "http://[::1]:8888/",
    "http://[::ffff:127.0.0.1]/a.jpg",
    "http://100.64.0.1/a.jpg",
]


@pytest.mark.parametrize("url", SSRF_URLS)
def test_ssrf_blocked(client, url: str) -> None:
    response = client.post("/v1/analyze", json={"url": url})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN_TARGET"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://127.0.0.1:6379/_INFO", "ftp://x/a"])
def test_bad_schemes_blocked(client, url: str) -> None:
    response = client.post("/v1/analyze", json={"url": url})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_URL"


# --------------------------------------------------------------------------
#  Path traversal
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["../../etc/passwd", "/etc/shadow", "/root/.ssh/id_rsa", "a/../../../etc/hosts"]
)
def test_path_traversal_blocked(client, path: str) -> None:
    response = client.post("/v1/analyze", json={"path": path})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "PATH_NOT_ALLOWED"


def test_path_root_allowlist(tmp_path, monkeypatch) -> None:
    """Oq ro'yxat ichidagi fayl o'qiladi, tashqarisidagi — yo'q."""
    from app.core import loader
    from tests.conftest import make_image

    inside = tmp_path / "ok.jpg"
    inside.write_bytes(make_image())
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(make_image())

    monkeypatch.setattr(loader._settings, "allowed_path_roots", [tmp_path.resolve()])

    assert loader.load_from_path("ok.jpg")[:3] == b"\xff\xd8\xff"

    from app.envelope import ApiError

    with pytest.raises(ApiError):
        loader.load_from_path("../outside.jpg")


def test_symlink_escape_blocked(tmp_path, monkeypatch) -> None:
    """Oq ro'yxat ichidagi symlink tashqariga ishora qilsa — rad etiladi."""
    from app.core import loader
    from app.envelope import ApiError
    from tests.conftest import make_image

    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(make_image())
    (root / "link.jpg").symlink_to(secret)

    monkeypatch.setattr(loader._settings, "allowed_path_roots", [root.resolve()])
    with pytest.raises(ApiError):
        loader.load_from_path("link.jpg")


# --------------------------------------------------------------------------
#  Kirish limitlari
# --------------------------------------------------------------------------


def test_non_image_rejected(client) -> None:
    payload = base64.b64encode(b"hello world, this is not an image").decode()
    response = client.post("/v1/analyze", json={"image_base64": payload})
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_corrupt_image_rejected(client) -> None:
    """To'g'ri JPEG imzosi, lekin buzuq tana."""
    payload = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 500).decode()
    response = client.post("/v1/analyze", json={"image_base64": payload})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "DECODE_FAILED"


def test_oversized_rejected(client, monkeypatch) -> None:
    from app.core import imaging
    from tests.conftest import make_image

    monkeypatch.setattr(imaging._settings, "max_image_bytes", 100)
    payload = base64.b64encode(make_image()).decode()
    response = client.post("/v1/analyze", json={"image_base64": payload})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_decompression_bomb_rejected() -> None:
    """Kichik fayl, ulkan piksel o'lchami — Pillow bomba himoyasi."""
    import io

    from PIL import Image

    from app.core import imaging
    from app.envelope import ApiError

    buffer = io.BytesIO()
    # Bir rangli katta PNG juda yaxshi siqiladi: ~30 KB fayl, 64 MP rasm.
    Image.new("RGB", (8000, 8000), (0, 0, 0)).save(buffer, format="PNG")
    data = buffer.getvalue()
    assert len(data) < imaging._settings.max_image_bytes

    with pytest.raises(ApiError) as exc:
        imaging.decode(data)
    assert exc.value.code == "IMAGE_TOO_LARGE_PIXELS"
