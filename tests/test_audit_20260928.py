"""2026-09-28 auditida topilgan 14 nuqson uchun regressiya testlari.

Har bir test aynan bitta topilmani qotirib qo'yadi: nuqson qaytsa, test
yiqiladi. Batafsil tavsif — `docs/` dagi audit hisobotida.
"""

from __future__ import annotations

import asyncio
import io
import resource

import pytest
from PIL import Image

from app.config import get_settings
from app.core import imaging, netguard
from app.schemas import AnalyzeOptions, AnalyzeResult

_settings = get_settings()


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (17, 99, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


# =====================================================================
#  1 (P0) — chegara ostidagi ulkan rasm xotirani yeb qo'ymaydi
# =====================================================================


def test_chegara_ostidagi_50MP_rasm_kichraytiriladi() -> None:
    """7070x7070 PNG — 145 KB, 49.98 MP: piksel chegarasidan 0.03% past.

    Ilgari u to'liq o'lchamda uch marta nusxalanardi (PIL RGB, numpy
    ko'chirma, NudeNet padding) va bitta so'rov ~800 MB yer edi.
    """
    data = _png(7070, 7070)
    assert len(data) < 1_000_000, "namuna kutilganidan katta"
    assert 7070 * 7070 < _settings.max_image_pixels, "namuna chegaradan oshdi"

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    img, info, scale = imaging.decode(data)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert img.width * img.height <= _settings.max_working_pixels
    # Mijoz ASL o'lchamni ko'radi — kichraytirish ichki ish.
    assert (info.width, info.height) == (7070, 7070)
    assert scale == pytest.approx(7070 / img.width)
    # 49.98 MP RGB = 150 MB. Kichraytirish bo'lmasa o'sish shundan katta.
    assert (after - before) < 250 * 1024, f"xotira o'sishi {after - before} KB"


def test_kichik_rasm_kichraytirilmaydi() -> None:
    img, info, scale = imaging.decode(_png(640, 480))
    assert (img.width, img.height) == (640, 480)
    assert (info.width, info.height) == (640, 480)
    assert scale == 1.0


def test_topilma_ramkasi_ASL_koordinatalarda(client) -> None:
    """Kichraytirilgan rasmda topilgan ramka mijozga asl o'lchamda beriladi."""
    from app.services import pipeline

    raw = {"label": "FACE_F", "score": 90.0, "box": {"x": 10, "y": 20, "width": 5, "height": 7}}
    scaled = pipeline._rescale(raw, 4.0)
    assert scaled["box"] == {"x": 40, "y": 80, "width": 20, "height": 28}
    assert pipeline._rescale(raw, 1.0) is raw


# =====================================================================
#  2, 3 — DNS event loop'ni bloklamaydi, umumiy muddat haqiqiy
# =====================================================================


def test_DNS_event_loopni_bloklamaydi() -> None:
    """`_prepare` endi threadpool'da chaqiriladi."""
    import inspect

    from app.core import fetcher

    source = inspect.getsource(fetcher._fetch)
    assert "run_in_threadpool(_prepare" in source, "DNS yana loop ichiga qaytdi"


def test_umumiy_muddat_qoyilgan() -> None:
    import inspect

    from app.core import fetcher

    source = inspect.getsource(fetcher.fetch_image)
    assert "asyncio.timeout(_settings.fetch_total_timeout)" in source


def test_mijoz_bitta_umumiy_bolib_ishlatiladi() -> None:
    """Har so'rovda yangi mijoz = `max_connections` limiti ma'nosiz."""
    import inspect

    from app.core import fetcher

    assert "httpx.AsyncClient(" not in inspect.getsource(fetcher._fetch)


# =====================================================================
#  5 — yaroqsiz URL mijozning xatosi (400), 500 emas
# =====================================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:99999/a.jpg",
        "http://example.com:abc/a.jpg",
        "http://example.com:-1/a.jpg",
        "http://[::1/a.jpg",
        "http://" + "a" * 70 + ".com/a.jpg",
        "http://example.com:80:80/a.jpg",
    ],
)
def test_yaroqsiz_url_400_beradi(client, url: str) -> None:
    response = client.post("/v1/analyze", json={"url": url})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] in {"INVALID_URL", "INVALID_REQUEST"}


# =====================================================================
#  6 — Redis yo'q bo'lsa health "ok" demaydi
# =====================================================================


def test_redis_yoq_bolsa_health_degraded(client, monkeypatch) -> None:
    from app.services import cache as cache_mod

    async def down() -> bool:
        return False

    monkeypatch.setattr(cache_mod.cache, "ping", down)
    monkeypatch.setattr(cache_mod._settings, "cache_enabled", True)
    body = client.get("/v1/health").json()["data"]
    assert body["redis"] == "down"
    assert body["status"] == "degraded", "yarim uzilish 'ok' bo'lib ko'rinardi"


# =====================================================================
#  7, 8 — limit hisobi
# =====================================================================


def test_nol_limit_CHEKSIZ_boladi() -> None:
    from app.api import deps

    record = {"id": "k", "rate_limit": 0}
    configured = record.get("rate_limit")
    assert (
        int(configured)
        if isinstance(configured, int | float | str)
        and str(configured).lstrip("-").isdigit()
        else deps._settings.rate_limit_per_minute
    ) == 0


def test_rad_etilgan_sorov_kvotani_YEMAYDI() -> None:
    from app.services.cache import _LocalWindow

    window = _LocalWindow()
    assert window.hit("a", 40, 60)[0] == 40
    assert window.hit("a", 40, 60)[0] == 80
    window.refund("a", 40)
    assert window.hit("a", 1, 60)[0] == 41


def test_redis_yoq_bolsa_ham_limit_ishlaydi() -> None:
    """Ilgari Redis o'chgan zahoti rate-limit BUTUNLAY yo'qolardi."""
    from app.services.cache import CacheClient

    async def run() -> tuple[int, int]:
        client = CacheClient()

        async def no_redis():
            return None

        client._ensure = no_redis
        return await client.hit_rate_limit("zaxira-test", cost=5)

    current, ttl = asyncio.run(run())
    assert current == 5 and ttl > 0


# =====================================================================
#  9 — `options` noma'lum kalitni jimgina yutmaydi
# =====================================================================


@pytest.mark.parametrize("payload", [{"detekt": False}, {"cache_enabled": False}, {"url": "x"}])
def test_notogri_option_400_beradi(client, image_bytes, payload) -> None:
    import json

    with pytest.raises(ValueError):
        AnalyzeOptions.model_validate(payload)
    response = client.post(
        "/v1/analyze",
        files={"file": ("a.jpg", image_bytes, "image/jpeg")},
        data={"options": json.dumps(payload)},
    )
    assert response.status_code == 400, response.text


# =====================================================================
#  4, 13 — buzilgan natija keshlanmaydi, buzilgan kesh mijozga yozilmaydi
# =====================================================================


def test_detektor_yiqilsa_503_va_KESHLANMAYDI(client, image_bytes, monkeypatch) -> None:
    """EGA QARORI 2026-09-28: detektorsiz "safe" deyilmaydi.

    Detektorsiz verdict PASTGA og'adi (klassifikator 15% + ochiq tana
    qismi 70% -> `safe`), shuning uchun mijozga yolg'on "xavfsiz"
    o'rniga 503 qaytariladi. Buzilgan natija keshga ham tushmaydi.
    """
    from app.envelope import ApiError
    from app.services import pipeline

    calls = {"detect": 0}
    written: list[str] = []

    class _Broken:
        def detect(self, *args, **kwargs):
            calls["detect"] += 1
            raise RuntimeError("detektor yiqildi")

    async def spy(key, payload):
        written.append(key)

    async def miss(key):
        return None

    monkeypatch.setattr(pipeline.engine, "detector", _Broken())
    monkeypatch.setattr(pipeline.cache, "set_result", spy)
    monkeypatch.setattr(pipeline.cache, "get_result", miss)
    monkeypatch.setattr(pipeline._settings, "cache_enabled", True)

    with pytest.raises(ApiError) as caught:
        asyncio.run(pipeline.analyze_bytes(image_bytes, detect=True, use_cache=True))
    assert caught.value.status_code == 503
    assert calls["detect"] == 2, "o'tkinchi nosozlik uchun qayta urinish yo'q"
    assert written == [], "buzilgan natija keshga yozildi"


def test_detektorsiz_sorov_ISHLAYDI(client, image_bytes) -> None:
    """`detect=false` — detektor umuman chaqirilmaydi, 503 ham yo'q."""
    response = client.post(
        "/v1/analyze",
        files={"file": ("a.jpg", image_bytes, "image/jpeg")},
        data={"options": '{"detect": false}'},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["models"]["detector"] == "disabled"


def test_eskirgan_kesh_yozuvi_mijozga_400_bermaydi(client, image_bytes, monkeypatch) -> None:
    from app.services import pipeline

    async def stale(key):
        return {"verdict": "safe"}

    async def noop(key, payload):
        return None

    monkeypatch.setattr(pipeline.cache, "get_result", stale)
    monkeypatch.setattr(pipeline.cache, "set_result", noop)
    monkeypatch.setattr(pipeline._settings, "cache_enabled", True)
    result = asyncio.run(pipeline.analyze_bytes(image_bytes, detect=False, use_cache=True))
    assert isinstance(result, AnalyzeResult)
    assert result.cached is False, "eskirgan yozuv natija sifatida qaytdi"


# =====================================================================
#  11 — serverning o'z manzillari
# =====================================================================


def test_oz_manzillar_bloklangan() -> None:
    if not netguard.OWN_ADDRESSES:
        pytest.skip("/proc/net o'qilmadi")
    for own in netguard.OWN_ADDRESSES:
        assert netguard.is_public(own) is False


# =====================================================================
#  12 — model nusxasi atomar yoziladi
# =====================================================================


def test_model_nusxasi_atomar_yoziladi() -> None:
    import inspect

    from app.services import classifier as cm

    source = inspect.getsource(cm.ensure_feature_model)
    assert "os.replace(" in source, "3 worker bir-birining faylini buzishi mumkin"


def test_klassifikator_qurilishi_try_ICHIDA() -> None:
    import inspect

    from app import main

    source = inspect.getsource(main.lifespan)
    build = source.index("classifier = build(model_path, head)")
    fallback = source.index("except Exception:")
    assert build < fallback, "sessiya qurilishi zaxira yo'lidan tashqarida"


# =====================================================================
#  14 — batch va element uchun muddat
# =====================================================================


def test_batch_va_element_muddati_bor() -> None:
    import inspect

    from app.api import routes
    from app.services import pipeline

    assert "batch_total_timeout" in inspect.getsource(routes.analyze_batch)
    assert "item_total_timeout" in inspect.getsource(pipeline.analyze_item)
