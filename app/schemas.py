"""So'rov va javob modellari (Pydantic v2)."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Verdict(str, Enum):
    SAFE = "safe"
    SUGGESTIVE = "suggestive"
    NSFW = "nsfw"
    NSFL = "nsfl"


# --------------------------------------------------------------------------
#  So'rovlar
# --------------------------------------------------------------------------


class AnalyzeOptions(BaseModel):
    """Barcha kirish rejimlarida umumiy ixtiyoriy parametrlar."""

    detect: bool = Field(
        default=True,
        description="Tana qismlarini aniqlovchi detektorni ishlatish (bbox bilan).",
    )
    min_detection_score: Annotated[float, Field(ge=0, le=100)] = Field(
        default=25.0,
        description="Bundan past ishonchli topilmalar javobga kiritilmaydi (%).",
    )
    cache: bool = Field(
        default=True,
        description="Rasm sha256 bo'yicha natijani keshdan o'qish/keshga yozish.",
    )


class AnalyzeRequest(AnalyzeOptions):
    """`POST /v1/analyze` uchun JSON tanasi.

    To'rt kirish rejimidan ANIQ BITTASI berilishi kerak.
    """

    model_config = ConfigDict(extra="forbid")

    url: str | None = Field(default=None, description="Rasm URL manzili (http/https).")
    path: str | None = Field(
        default=None,
        description="Serverdagi fayl yo'li — faqat oq ro'yxatdagi kataloglar ichida.",
    )
    image_base64: str | None = Field(
        default=None, description="Base64 rasm (data-URI prefiksi ham qabul qilinadi)."
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> AnalyzeRequest:
        provided = [
            name
            for name, value in (
                ("url", self.url),
                ("path", self.path),
                ("image_base64", self.image_base64),
            )
            if value
        ]
        if len(provided) != 1:
            raise ValueError(
                "Aniq bitta kirish manbasi berilishi kerak: url, path yoki image_base64"
                + (f" (berilgan: {', '.join(provided)})" if provided else "")
            )
        return self


class BatchItem(AnalyzeRequest):
    """Batch elementi — `id` mijoz tomonidan javobni moslashtirish uchun."""

    id: str | None = Field(
        default=None, description="Javobda o'zgarishsiz qaytariladigan mijoz identifikatori."
    )


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[BatchItem] = Field(min_length=1)


# --------------------------------------------------------------------------
#  Javoblar
# --------------------------------------------------------------------------


class Box(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Detection(BaseModel):
    label: str = Field(description="Masalan: FEMALE_BREAST_EXPOSED")
    score: float = Field(description="Ishonch, foizda (0-100).")
    box: Box


class Scores(BaseModel):
    """Barcha ballar foizda (0.00-100.00), 2 kasr raqamgacha yaxlitlangan."""

    sfw: float
    suggestive: float
    nsfw: float
    nsfl: float


class ImageInfo(BaseModel):
    width: int
    height: int
    format: str
    size_bytes: int
    sha256: str


class Timings(BaseModel):
    fetch: int = 0
    decode: int = 0
    classify: int = 0
    detect: int = 0
    total: int = 0


class AnalyzeResult(BaseModel):
    verdict: Verdict
    is_nsfw: bool
    is_safe: bool
    needs_review: bool = Field(
        default=False,
        description=(
            "Qaror chegaraviy zonada chiqdi — qo'lda ko'rib chiqishga arziydi. "
            "`is_safe` ga ta'sir qilmaydi: bloklash qarori mijozniki."
        ),
    )
    confidence: float = Field(description="Chiqarilgan verdict ballining foizi.")
    scores: Scores
    detections: list[Detection] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    image: ImageInfo
    models: dict[str, str]
    timings_ms: Timings
    cached: bool = False


class BatchResultItem(BaseModel):
    id: str | None = None
    index: int
    success: bool
    data: AnalyzeResult | None = None
    error: dict[str, Any] | None = None


class BatchResult(BaseModel):
    count: int
    succeeded: int
    failed: int
    results: list[BatchResultItem]


class HealthResult(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    models_loaded: bool
    redis: Literal["ok", "down", "disabled"]
    uptime_s: int


class ModelsResult(BaseModel):
    classifier: dict[str, Any]
    detector: dict[str, Any]
    limits: dict[str, Any]
    thresholds: dict[str, float]
