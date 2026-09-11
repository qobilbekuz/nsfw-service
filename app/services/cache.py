"""Redis: natija keshi, API kalitlari va rate-limit.

Redis o'chib qolsa servis ISHLASHDA DAVOM ETADI — kesh va rate-limit
"fail-open" rejimida (auth esa fail-closed, §4.4 ga qarang).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from app.config import get_settings

log = logging.getLogger("nsfw.cache")
_settings = get_settings()

KEY_PREFIX = "nsfw:"
RESULT_KEY = KEY_PREFIX + "res:{}"
APIKEY_KEY = KEY_PREFIX + "key:{}"
RATE_KEY = KEY_PREFIX + "rl:{}:{}"

# Sliding-window emas, sobit oyna: bitta INCR + EXPIRE atomar bajariladi.
# EXPIRE faqat birinchi INCR dan keyin qo'yiladi, aks holda oyna cheksiz
# uzayib ketardi ("sliding TTL" muammosi).
_RATE_LUA = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
return {current, ttl}
"""


class CacheClient:
    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None
        self._rate_script = None
        self.available = False

    async def connect(self) -> None:
        try:
            self._redis = aioredis.from_url(
                _settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            )
            await self._redis.ping()
            self._rate_script = self._redis.register_script(_RATE_LUA)
            self.available = True
        except Exception as exc:  # noqa: BLE001 — Redis yo'qligi fatal emas
            log.warning("Redis ulanmadi (%s) — kesh va rate-limit o'chirildi", exc)
            self._redis = None
            self.available = False

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            self.available = False

    async def ping(self) -> bool:
        if self._redis is None:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:  # noqa: BLE001
            return False

    # --- Natija keshi -----------------------------------------------------

    async def get_result(self, sha256: str) -> dict[str, Any] | None:
        if self._redis is None or not _settings.cache_enabled:
            return None
        try:
            raw = await self._redis.get(RESULT_KEY.format(sha256))
        except Exception:  # noqa: BLE001
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_result(self, sha256: str, payload: dict[str, Any]) -> None:
        if self._redis is None or not _settings.cache_enabled:
            return
        try:
            await self._redis.set(
                RESULT_KEY.format(sha256),
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

    async def lookup_key(self, api_key: str) -> dict[str, Any] | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(APIKEY_KEY.format(self.hash_key(api_key)))
        except Exception:  # noqa: BLE001
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    # --- Rate limit -------------------------------------------------------

    async def hit_rate_limit(self, subject: str, window_s: int = 60) -> tuple[int, int]:
        """`(joriy_soni, ttl)` qaytaradi. Redis yo'q bo'lsa `(0, window_s)`."""
        if self._redis is None or self._rate_script is None:
            return 0, window_s
        try:
            bucket = RATE_KEY.format(subject, window_s)
            current, ttl = await self._rate_script(keys=[bucket], args=[window_s])
            return int(current), int(ttl if ttl > 0 else window_s)
        except Exception:  # noqa: BLE001
            return 0, window_s


cache = CacheClient()
