"""Tana qismlari detektori: NudeNet 3.4.2 (YOLOv8-n, 320x320, ONNX, MIT).

Klassifikator faqat "nechi %" beradi, lekin NIMA UCHUN ekanini tushuntirmaydi.
Detektor 18 ta sinfni bbox bilan qaytaradi — shu tufayli javob izohlanadigan
bo'ladi va `suggestive` (yopiq/yarim ochiq) darajasini ajratish mumkin.
"""

from __future__ import annotations

from pathlib import Path

import nudenet
import numpy as np
import onnxruntime as ort
from nudenet import NudeDetector

MODEL_NAME = "nudenet-320n"

# Ochiq-oydin 18+ ko'rsatkichlari — bittasi topilsa verdict `nsfw` bo'ladi.
EXPLICIT_LABELS = frozenset(
    {
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED",
    }
)

# Shahvoniy ishora: yopiq, lekin urg'u berilgan joylar (bikini, ich kiyim).
SUGGESTIVE_LABELS = frozenset(
    {
        "FEMALE_GENITALIA_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
        "ANUS_COVERED",
        "BELLY_EXPOSED",
        "ARMPITS_EXPOSED",
    }
)

# Qolganlari (FACE_*, FEET_*, BELLY_COVERED, ...) verdictga ta'sir qilmaydi.


class BodyPartDetector:
    def __init__(self, intra_threads: int = 2, inter_threads: int = 1) -> None:
        self._detector = NudeDetector()

        # NudeNet o'z sessiyasini thread sozlamalarisiz yaratadi — u holda ORT
        # 20 yadroning hammasini egallaydi va 3 uvicorn worker bir-birini
        # bo'g'adi. Sessiyani boshqariladigan variant bilan almashtiramiz.
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_threads
        options.inter_op_num_threads = inter_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.model_path = self._model_file()
        self._detector.onnx_session = ort.InferenceSession(
            str(self.model_path), options, providers=["CPUExecutionProvider"]
        )
        self._detector.input_name = self._detector.onnx_session.get_inputs()[0].name

    @staticmethod
    def _model_file() -> Path:
        return Path(nudenet.__file__).parent / "320n.onnx"

    def detect(self, bgr: np.ndarray, min_score: float) -> list[dict[str, object]]:
        """BGR numpy massivdan topilmalar ro'yxatini qaytaradi.

        `min_score` foizda (0-100). Chiqishdagi `score` ham foizda.
        """
        raw = self._detector.detect(bgr)
        out: list[dict[str, object]] = []
        for item in raw:
            score = float(item["score"]) * 100.0
            if score < min_score:
                continue
            x, y, w, h = item["box"]
            out.append(
                {
                    "label": item["class"],
                    "score": round(score, 2),
                    "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                }
            )
        out.sort(key=lambda d: d["score"], reverse=True)
        return out

    def info(self) -> dict[str, object]:
        path = self.model_path
        return {
            "name": MODEL_NAME,
            "source": "notAI-tech/NudeNet@3.4.2",
            "architecture": "YOLOv8-n",
            "license": "MIT",
            "input_size": 320,
            "classes": 18,
            "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
        }
