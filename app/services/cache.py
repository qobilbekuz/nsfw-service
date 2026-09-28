"""Redis: natija keshi, API kalitlari va rate-limit.

Redis o'chib qolsa servis ISHLASHDA DAVOM ETADI — kesh va rate-limit
"fail-open" rejimida (auth esa fail-closed, §4.4 ga qarang; zaxira kalitlar
uchun `deps.py` va `BOOTSTRAP_KEYS`).

QAYTA ULANISH (2026-09-25): ilgari ulanish faqat lifespan'da bir marta
urinilardi. Servis Redis'dan OLDIN ko'tarilsa (reboot poygasi) `_redis`
restartgacha `None` bo'lib qolardi: kesh ham, rate-limit ham o'chiq, API
kalitlari esa hammasi 401. Endi har bir chaqiruv `_ensure()` orqali o'tadi va
ulanish yo'q bo'lsa `redis_retry_seconds` da bir qayta uriniladi.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis

from app.config import get_settings

log = logging.getLogger("nsfw.cache")
_settings = get_settings()

KEY_PREFIX = "nsfw:"
# Natija kaliti pipeline.py da yasaladi va model/bosh/chegaralar versiyasini
# (fingerprint) o'z ichiga oladi — busiz qayta kalibrlashdan keyin eski
# verdictlar yana 7 kun keshdan qaytardi.
RESULT_KEY = KEY_PREFIX + "res:{}"
APIKEY_KEY = KEY_PREFIX + "key:{}"
RATE_KEY = KEY_PREFIX + "rl:{}:{}"

# Sobit oyna: INCRBY + EXPIRE atomar. EXPIRE faqat oynaning birinchi
# urilishida qo'yiladi ("sliding TTL" muammosi bo'lmasin). `ARGV[2]` — narx:
# batch so'rovi rasm soniga teng birlik yeydi (ilgari 20 rasm = 1 birlik edi).
# TTL yo'qolgan kalit (masalan qo'lda SET qilingan) abadiy qolmasligi uchun
# `ttl < 0` holati ham tuzatiladi.
_RATE_LUA = """
local current = redis.call('INCRBY', KEYS[1], ARGV[2])
if current == tonumber(ARGV[2]) then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
if ttl < 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
  ttl = tonumber(ARGV[1])
end
return {current, ttl}
"""

#: `lookup_key` holati: kalit topildi / Redis'da yo'q / Redis'ga yetib bo'lmadi.
FOUND, MISSING, UNAVAILABLE = "found", "missing", "unavailable"


class _LocalWindow:
    """Redis yo'q paytdagi zaxira hisoblagich (shu jarayon ichida).

    Ilgari Redis o'chgan zahoti rate-limit BUTUNLAY yo'qolardi. Bu zaxira
    aniq emas (har worker'da alohida), lekin cheksiz oqimni to'xtatadi.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, int]] = {}

    def hit(self, subject: str, cost: int, window_s: int) -> tuple[int, int]:
        now = time.monotonic()
        start, used = self._buckets.get(subject, (now, 0))
        if now - start >= window_s:
            start, used = now, 0
        used += cost
        self._buckets[subject] = (start, used)
        if len(self._buckets) > 10_000:
            self._buckets = {
                key: value
                for key, value in self._buckets.items()
                if now - value[0] < window_s
            }
        return used, max(1, int(window_s - (now - start)))

    def refund(self, subject: str, cost: int) -> None:
        entry = self._buckets.get(subject)
        if entry is not None:
            self._buckets[subject] = (entry[0], max(0, entry[1] - cost))


class CacheClient:
    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None
        self._rate_script = None
        self._last_attempt = float("-inf")
        self._local = _LocalWindow()
        self.available = False

    async def connect(self) -> None:
        self._last_attempt = time.monotonic()
        try:
            client = aioredis.from_url(
                _settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            )
            await client.ping()
            self._redis = client
            self._rate_script = client.register_script(_RATE_LUA)
            if not self.available:
                log.info("Redis ulandi")
            self.available = True
        except Exception as exc:  # noqa: BLE001 — Redis yo'qligi fatal emas
            log.warning(
                "Redis ulanmadi (%s) — %ss dan keyin qayta uriniladi",
                exc, _settings.redis_retry_seconds,
            )
            self._redis = None
            self._rate_script = None
            self.available = False

    async def _ensure(self) -> aioredis.Redis | None:
        if self._redis is not None:
            return self._redis
        if time.monotonic() - self._last_attempt >= _settings.redis_retry_seconds:
            await self.connect()
        return self._redis

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            self.available = False

    async def ping(self) -> bool:
        client = await self._ensure()
        if client is None:
            return False
        try:
            await client.ping()
            return True
        except Exception:  # noqa: BLE001
            return False

    # --- Natija keshi -----------------------------------------------------

    async def get_result(self, key: str) -> dict[str, Any] | None:
        if not _settings.cache_enabled:
            return None
        client = await self._ensure()
        if client is None:
            return None
        try:
            raw = await client.get(RESULT_KEY.format(key))
        except Exception:  # noqa: BLE001
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_result(self, key: str, payload: dict[str, Any]) -> None:
        if not _settings.cache_enabled:
            return
        client = await self._ensure()
        if client is None:
            return
        try:
            await client.set(
                RESULT_KEY.format(key),
                json.dumps(payload, separators=(",", ":")),
                ex=_settings.cache_ttl_seconds,
            )
        except Exception:  # noqa: BLE001
            pass

    # --- API kalitlari ----------------------------------------------------

    @staticmethod
    def hash_key(api_key: str) -> str:
        """Kalit Redis'da ochiq matnda saqlanmaydi — faqat sha256 dayjesti."""
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    async def lookup_key(self, api_key: str) -> tuple[str, dict[str, Any] | None]:
        """`(holat, yozuv)`: holat — FOUND / MISSING / UNAVAILABLE.

        "Redis'da yo'q" va "Redis'ga yetib bo'lmadi" endi FARQLANADI — `deps.py`
        ikkala holatda ham zaxira kalitlarga (BOOTSTRAP_KEYS) qaraydi, lekin
        logda sabab aniq ko'rinadi.
        """
        client = await self._ensure()
        if client is None:
            return UNAVAILABLE, None
        try:
            raw = await client.get(APIKEY_KEY.format(self.hash_key(api_key)))
        except Exception:  # noqa: BLE001
            return UNAVAILABLE, None
        if not raw:
            return MISSING, None
        try:
            return FOUND, json.loads(raw)
        except json.JSONDecodeError:
            return MISSING, None

    # --- Rate limit -------------------------------------------------------

    async def hit_rate_limit(
        self, subject: str, cost: int = 1, window_s: int = 60
    ) -> tuple[int, int]:
        """`(joriy_soni, ttl)`. Redis yo'q bo'lsa jarayon ichidagi zaxira."""
        cost = max(1, int(cost))
        client = await self._ensure()
        if client is not None and self._rate_script is not None:
            try:
                bucket = RATE_KEY.format(subject, window_s)
                current, ttl = await self._rate_script(
                    keys=[bucket], args=[window_s, cost]
                )
                return int(current), int(ttl if ttl > 0 else window_s)
            except Exception:  # noqa: BLE001
                pass
        return self._local.hit(subject, cost, window_s)

    async def refund_rate_limit(
        self, subject: str, cost: int = 1, window_s: int = 60
    ) -> None:
        """Rad etilgan so'rov hisobdan qaytariladi.

        Lua skripti chegarani tekshirishdan OLDIN `INCRBY` qiladi, ya'ni
        429 olgan mijoz o'ziga berilmagan kvotani ham yeb qo'yardi.
        """
        cost = max(1, int(cost))
        self._local.refund(subject, cost)
        client = await self._ensure()
        if client is None:
            return
        try:
            await client.incrby(RATE_KEY.format(subject, window_s), -cost)
        except Exception:  # noqa: BLE001
            pass


cache = CacheClient()
