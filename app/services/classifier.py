"""Asosiy klassifikator: OwenElliott/image-safety-classifier-s (SwiftFormer-S).

Model: 6.1M parametr, ONNX 23 MB, MIT litsenziya, ~7 ms/rasm (CPU).
Uch sinf: NSFL (qon/zo'ravonlik), NSFW (18+), SFW.

MUHIM: normalizatsiya va softmax ONNX grafigi ICHIGA pishirilgan. Ya'ni
kirish — oddiy 0-255 oraliqdagi float32 NCHW tenzor, chiqish — tayyor
ehtimolliklar. Bu yerda qo'shimcha normalize QILMASLIK kerak.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# ONNX chiqishidagi kanallar tartibi (model config.json → label_names).
LABELS = ("nsfl", "nsfw", "sfw")
INPUT_SIZE = 224
MODEL_NAME = "image-safety-classifier-s"

#: Backbone chiqishi — oxirgi `ReduceMean` tugunining (1, 224) tenzori.
#: Ikkala klassifikatsiya boshi (`model.head`, `model.head_dist`) shundan
#: oziqlanadi, ya'ni bu rasm haqidagi sinflarga bo'linishdan oldingi ma'lumot.
EMBEDDING_TENSOR = "mean_1"
EMBEDDING_DIM = 224

log = logging.getLogger("nsfw.classifier")


def ensure_feature_model(src: Path, dst: Path) -> Path:
    """`mean_1` ni chiqishga qo'shilgan model nusxasini yaratadi (kerak bo'lsa).

    Graf o'zgarmaydi — mavjud oraliq tenzor chiqish deb belgilanadi, xolos.
    Shu sababli `probabilities` bit-ma-bit o'sha-o'sha qoladi va qo'shimcha
    hisoblash xarajati yo'q (tenzor baribir hisoblanardi).
    """
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst

    import onnx  # faqat shu yerda kerak — oddiy ish rejimida import qilinmaydi

    model = onnx.load(str(src))
    if EMBEDDING_TENSOR not in {o.name for o in model.graph.output}:
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                EMBEDDING_TENSOR, onnx.TensorProto.FLOAT, [1, EMBEDDING_DIM]
            )
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    log.info("embedding chiqishli model yaratildi: %s", dst.name)
    return dst


def load_custom_head(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """`train_head.py` saqlagan og'irliklarni o'qiydi va tekshiradi."""
    blob = np.load(path, allow_pickle=True)
    W = blob["W"].astype(np.float32)
    b = blob["b"].astype(np.float32)

    if W.shape != (EMBEDDING_DIM, len(LABELS)) or b.shape != (len(LABELS),):
        raise ValueError(
            f"{path.name}: kutilgan shakl W{(EMBEDDING_DIM, len(LABELS))} "
            f"b{(len(LABELS),)}, topilgani W{W.shape} b{b.shape}"
        )
    # Sinflar tartibi mos kelmasa ballar jimgina almashib ketardi.
    classes = tuple(str(c) for c in blob["classes"])
    if classes != LABELS:
        raise ValueError(f"{path.name}: sinflar tartibi {classes}, kutilgani {LABELS}")
    return W, b


class SafetyClassifier:
    def __init__(
        self,
        model_path: Path,
        intra_threads: int = 2,
        inter_threads: int = 1,
        custom_head: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_threads
        options.inter_op_num_threads = inter_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Modelni yuklashda ORT ba'zi ogohlantirishlarni chiqaradi; loglarni
        # toza saqlash uchun faqat xatolarni ko'rsatamiz.
        options.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = model_path
        self.custom_head = custom_head

        if custom_head is not None:
            outputs = {o.name for o in self.session.get_outputs()}
            if EMBEDDING_TENSOR not in outputs:
                raise RuntimeError(
                    f"{model_path.name} da '{EMBEDDING_TENSOR}' chiqishi yo'q — "
                    "o'z boshingizni ishlatish uchun ensure_feature_model() bilan "
                    "tayyorlangan model kerak"
                )
            log.info("o'z boshingiz ulandi (asl bosh chetlab o'tiladi)")

    def preprocess(self, img: Image.Image) -> np.ndarray:
        resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32)  # HWC, 0-255
        return np.transpose(array, (2, 0, 1))[None, ...]  # NCHW

    def predict(self, img: Image.Image) -> dict[str, float]:
        """Foizdagi (0-100) ehtimolliklarni qaytaradi: `{nsfl, nsfw, sfw}`."""
        tensor = self.preprocess(img)

        if self.custom_head is None:
            probs = self.session.run(None, {self.input_name: tensor})[0][0]
        else:
            # Asl bosh o'rniga o'zimizniki. Backbone o'sha-o'sha, shuning uchun
            # grafga pishirilgan normalizatsiya ham o'z kuchida qoladi.
            W, b = self.custom_head
            emb = self.session.run(
                [EMBEDDING_TENSOR], {self.input_name: tensor}
            )[0][0]
            logits = emb @ W + b
            logits -= logits.max()
            exp = np.exp(logits)
            probs = exp / exp.sum()

        return {label: float(probs[i]) * 100.0 for i, label in enumerate(LABELS)}

    def info(self) -> dict[str, object]:
        return {
            "name": MODEL_NAME,
            "source": "OwenElliott/image-safety-classifier-s",
            "architecture": "SwiftFormer-S",
            "license": "MIT",
            "input_size": INPUT_SIZE,
            "classes": list(LABELS),
            "size_mb": round(self.model_path.stat().st_size / 1024 / 1024, 1),
            "head": "custom" if self.custom_head is not None else "original",
        }
