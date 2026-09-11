"""O'z boshini ulash mexanizmi (training/train_head.py natijasi).

Eng katta xavf — jimgina noto'g'ri ishlash: bosh yuklanadi, lekin sinflar
tartibi boshqacha bo'ladi va `nsfw` bilan `sfw` ballari almashib ketadi.
Shuning uchun tekshiruvlar qattiq va xato ANIQ ko'tariladi.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import get_settings
from app.core import imaging
from app.services import classifier as cm


@pytest.fixture(scope="module")
def feature_model(tmp_path_factory) -> Path:
    settings = get_settings()
    dst = tmp_path_factory.mktemp("models") / "features.onnx"
    return cm.ensure_feature_model(settings.classifier_path, dst)


def _head(path: Path, *, dim: int = cm.EMBEDDING_DIM, classes=cm.LABELS) -> Path:
    np.savez(
        path,
        W=np.zeros((dim, len(cm.LABELS)), np.float32),
        b=np.zeros(len(cm.LABELS), np.float32),
        classes=np.array(classes),
    )
    return path


def test_feature_model_exposes_embedding(feature_model: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(feature_model), providers=["CPUExecutionProvider"])
    outputs = {o.name: o.shape for o in session.get_outputs()}
    assert cm.EMBEDDING_TENSOR in outputs
    assert outputs[cm.EMBEDDING_TENSOR][-1] == cm.EMBEDDING_DIM
    assert "probabilities" in outputs  # asl chiqish yo'qolmagan bo'lishi shart


def test_feature_model_probabilities_identical(feature_model: Path, image_bytes) -> None:
    """Grafga chiqish qo'shish ehtimolliklarni O'ZGARTIRMASLIGI kerak.

    Agar bu test yiqilsa, o'qitish to'plamidagi ballar ishlab chiqarishdagidan
    farq qiladi va o'qitilgan bosh notog'ri taqsimotda o'qitilgan bo'ladi.
    """
    settings = get_settings()
    original = cm.SafetyClassifier(settings.classifier_path)
    with_features = cm.SafetyClassifier(feature_model)

    img, _info = imaging.decode(image_bytes)
    a = original.predict(img)
    b = with_features.predict(img)
    assert a == b


def test_ensure_feature_model_is_idempotent(feature_model: Path) -> None:
    settings = get_settings()
    before = feature_model.stat().st_mtime_ns
    again = cm.ensure_feature_model(settings.classifier_path, feature_model)
    assert again == feature_model
    assert feature_model.stat().st_mtime_ns == before  # qayta yozilmagan


def test_head_rejects_wrong_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kutilgan shakl"):
        cm.load_custom_head(_head(tmp_path / "h.npz", dim=64))


def test_head_rejects_wrong_class_order(tmp_path: Path) -> None:
    """Aynan shu xato ballarni jimgina almashtirib yuborardi."""
    with pytest.raises(ValueError, match="sinflar tartibi"):
        cm.load_custom_head(_head(tmp_path / "h.npz", classes=("sfw", "nsfw", "nsfl")))


def test_head_accepts_valid(tmp_path: Path) -> None:
    W, b = cm.load_custom_head(_head(tmp_path / "h.npz"))
    assert W.shape == (cm.EMBEDDING_DIM, len(cm.LABELS))
    assert b.shape == (len(cm.LABELS),)


def test_custom_head_requires_feature_model(tmp_path: Path) -> None:
    """Asl model bilan o'z boshini ishlatib bo'lmaydi — aniq xato kerak."""
    settings = get_settings()
    head = cm.load_custom_head(_head(tmp_path / "h.npz"))
    with pytest.raises(RuntimeError, match=cm.EMBEDDING_TENSOR):
        cm.SafetyClassifier(settings.classifier_path, custom_head=head)


def test_custom_head_changes_predictions(feature_model: Path, image_bytes) -> None:
    """Bosh haqiqatan ham ishlatilyaptimi (chetlab o'tilmayaptimi)."""
    img, _info = imaging.decode(image_bytes)

    # `nsfw` ni har doim ustun qiladigan sun'iy bosh.
    W = np.zeros((cm.EMBEDDING_DIM, len(cm.LABELS)), np.float32)
    b = np.zeros(len(cm.LABELS), np.float32)
    b[cm.LABELS.index("nsfw")] = 50.0

    model = cm.SafetyClassifier(feature_model, custom_head=(W, b))
    scores = model.predict(img)
    assert scores["nsfw"] > 99.0
    assert model.info()["head"] == "custom"

    baseline = cm.SafetyClassifier(feature_model)
    assert baseline.info()["head"] == "original"
    assert baseline.predict(img)["nsfw"] < 99.0
