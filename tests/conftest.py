"""Testlar uchun umumiy fixture'lar.

Testlar HAQIQIY modellar bilan ishlaydi (yuklanish ~200 ms) — mock qilmaymiz,
chunki asosiy xavf aynan preprocessing/kanal tartibi kabi joylarda.
"""

from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# DIQQAT: bu qatorlar `app.*` importlaridan OLDIN turishi shart —
# `get_settings()` lru_cache bilan keshlanadi va birinchi chaqiruvdagi
# muhit o'zgaruvchilari butun sessiya davomida saqlanib qoladi.
#
# TestClient mijoz IP'si sifatida "testclient" satrini beradi, u esa
# TRUSTED_IPS ga tushmaydi. Auth mantiqining o'zi test_api.py da emas,
# alohida tekshiriladi — bu yerda uni chetlab o'tamiz.
os.environ["PUBLIC_MODE"] = "true"
# Kesh o'chiriladi, aks holda bir test natijasi ikkinchisiga sizib o'tardi.
os.environ["CACHE_ENABLED"] = "false"
# Rate-limit testlarni tasodifiy yiqitmasin.
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"


@pytest.fixture(scope="session")
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def make_image(
    color: tuple[int, int, int] = (120, 140, 160),
    size: tuple[int, int] = (256, 256),
    fmt: str = "JPEG",
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def image_bytes() -> bytes:
    return make_image()
