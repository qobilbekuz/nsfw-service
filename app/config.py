"""Servis sozlamalari — .env dan o'qiladi (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Umumiy ---
    app_name: str = "NSFW Detection API"
    version: str = "1.0.0"
    root_path: str = "/nsfw"
    debug: bool = False

    # --- Modellar ---
    models_dir: Path = BASE_DIR / "models"
    classifier_file: str = "image-safety-classifier-s.onnx"
    # `training/train_head.py` o'qitgan bosh (masalan "custom_head.npz").
    # Bo'sh bo'lsa asl model boshi ishlatiladi. Backbone har ikki holda ham
    # o'zgarmaydi — faqat oxirgi chiziqli qatlam almashadi.
    custom_head_file: str = ""
    # NudeNet o'z modelini paket ichidan oladi; bu yerda faqat yoqish/o'chirish.
    detector_enabled: bool = True

    # onnxruntime thread'lari. 3 worker x 2 = 6 yadro; serverda boshqa
    # servislar ham ishlaydi, shuning uchun ataylab past.
    onnx_intra_threads: int = 2
    onnx_inter_threads: int = 1

    # --- Kirish limitlari ---
    max_image_bytes: int = 20 * 1024 * 1024  # 20 MB
    max_image_pixels: int = 50_000_000  # dekompressiya bombasiga qarshi
    max_batch_items: int = 20
    batch_concurrency: int = 8

    # --- URL yuklash (SSRF) ---
    fetch_connect_timeout: float = 5.0
    fetch_read_timeout: float = 10.0
    fetch_total_timeout: float = 15.0
    fetch_max_redirects: int = 3
    # Faqat sinov/ichki muhitda true qiling — privat IP'larga so'rovga ruxsat beradi.
    allow_private_targets: bool = False

    # --- Path rejimi ---
    # Vergul bilan ajratilgan absolyut kataloglar. Bo'sh => path rejimi o'chiq.
    # `NoDecode` — pydantic-settings ro'yxat turlarini .env dan JSON deb
    # o'qishga urinadi; u qadamni o'tkazib yuborib, pastdagi validatorga
    # oddiy `a,b,c` yozuvini qabul qilish imkonini beramiz.
    allowed_path_roots: Annotated[list[Path], NoDecode] = Field(default_factory=list)

    # --- Auth / limitlar ---
    public_mode: bool = False  # true => API kalitisiz ishlaydi
    trusted_ips: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["127.0.0.1", "::1"]
    )
    rate_limit_per_minute: int = 60

    # --- Redis ---
    redis_url: str = "redis://127.0.0.1:6379/4"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 7 * 24 * 3600

    # --- Skorlash chegaralari (foizda, 0-100) ---
    threshold_nsfw: float = 60.0
    threshold_nsfl: float = 50.0
    threshold_suggestive: float = 25.0
    # Klassifikator shu chegaradan yuqori bo'lsa, verdict detektor tasdig'isiz
    # ham `nsfw` bo'ladi. Oraliq zonada (threshold_nsfw..bu qiymat) esa
    # detektordan ochiq tana qismi topilishi talab qilinadi — sababi
    # `docs/TASKS.md` §1.2 va scoring.py izohida.
    threshold_nsfw_confident: float = 90.0
    # Detektor topilmalari uchun minimal ishonch
    detection_min_score: float = 25.0

    @field_validator("allowed_path_roots", "trusted_ips", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """`.env` da vergul bilan yozilgan ro'yxatni qabul qilish.

        pydantic-settings murakkab turlar uchun JSON kutadi; bu yerda oddiy
        `a,b,c` yozuvini ham qo'llab-quvvatlaymiz.
        """
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("["):  # JSON ko'rinishi — pydantic o'zi hal qilsin
                return v
            return [part.strip() for part in s.split(",") if part.strip()]
        return v

    @field_validator("allowed_path_roots")
    @classmethod
    def _resolve_roots(cls, v: list[Path]) -> list[Path]:
        return [p.resolve() for p in v]

    @property
    def classifier_path(self) -> Path:
        return self.models_dir / self.classifier_file

    @property
    def custom_head_path(self) -> Path | None:
        return self.models_dir / self.custom_head_file if self.custom_head_file else None

    @property
    def feature_model_path(self) -> Path:
        """Embedding chiqishi qo'shilgan model nusxasi (kerak bo'lsa yaratiladi)."""
        return self.models_dir / "classifier_with_features.onnx"


@lru_cache
def get_settings() -> Settings:
    return Settings()
