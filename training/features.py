"""Embedding chiqarish — frozen backbone yondashuvining poydevori.

SwiftFormer-S grafigi ichida oxirgi `ReduceMean` tuguni `mean_1` nomli
(1, 224) tenzor beradi; ikkala klassifikatsiya boshi (`model.head` va
`model.head_dist`) aynan shundan oziqlanadi. Ya'ni `mean_1` — backbone'ning
rasm haqidagi butun "fikri", sinflarga bo'linishdan oldingi holati.

Uni grafga chiqish sifatida qo'shsak, backbone'ni qayta o'qitmasdan turib
o'z klassifikatorimizni shu 224 o'lchovli vektor ustida o'qitishimiz mumkin
bo'ladi. PyTorch kerak emas — hammasi onnxruntime + numpy.

MUHIM: bu yerdagi preprocessing servisning `SafetyClassifier.preprocess`
metodidan olinadi. Ikkisi ajralib ketsa, o'qitilgan bosh ishlab chiqarishda
boshqa taqsimotni ko'radi va aniqlik jimgina tushadi.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# Graf jarrohligi servis tomonida yashaydi — o'qitish va ishlab chiqarish
# AYNAN bir xil modelni ko'rishi uchun. Bu yerda faqat qayta eksport qilamiz.
from app.services.classifier import (  # noqa: F401
    EMBEDDING_DIM,
    EMBEDDING_TENSOR,
    LABELS,
    SafetyClassifier,
    ensure_feature_model,
)


class FeatureExtractor:
    """Bitta o'tishda ham ehtimolliklarni, ham embeddingni qaytaradi."""

    def __init__(self, model_path: Path, threads: int = 2) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        outputs = {o.name for o in self.session.get_outputs()}
        if EMBEDDING_TENSOR not in outputs:
            raise RuntimeError(
                f"{model_path} da '{EMBEDDING_TENSOR}' chiqishi yo'q — "
                "avval ensure_feature_model() ni chaqiring"
            )

    # Preprocessing servis bilan bir xil bo'lishi uchun uning o'z metodini
    # ishlatamiz (SafetyClassifier nusxasi shart emas — metod holatga bog'liq emas).
    preprocess = SafetyClassifier.preprocess

    def run(self, img: Image.Image) -> tuple[dict[str, float], np.ndarray]:
        tensor = self.preprocess(img)
        probs, emb = self.session.run(
            ["probabilities", EMBEDDING_TENSOR], {self.input_name: tensor}
        )
        scores = {label: float(probs[0][i]) * 100.0 for i, label in enumerate(LABELS)}
        return scores, emb[0].astype(np.float32)
