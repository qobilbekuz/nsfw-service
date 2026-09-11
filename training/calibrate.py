#!/usr/bin/env python
"""Daraja 0: modelni o'qitmasdan, faqat chegaralarni to'g'rilash.

Ko'pincha eng arzon va eng katta yutuq shu yerda. `.env` dagi
`THRESHOLD_*` qiymatlari hozir taxminan qo'yilgan — bu skript ularni
sizning to'plamingizda o'lchab, eng yaxshi kombinatsiyani topadi.

MUHIM: bu yerda mantiq qayta yozilmagan. Servisning haqiqiy
`scoring.evaluate()` funksiyasi chaqiriladi, chegaralar esa vaqtincha
almashtiriladi. Shu sababli topilgan qiymatlarni `.env` ga ko'chirish
kifoya — natija aynan mos keladi.

Foydalanish:
    .venv/bin/python training/calibrate.py --data dataset.npz
    .venv/bin/python training/calibrate.py --data dataset.npz --min-recall 0.95
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import Box, Detection, Verdict  # noqa: E402
from app.services import detector as det  # noqa: E402
from app.services import scoring  # noqa: E402

# Sun'iy topilmalar: `evaluate()` faqat shu ikki to'plam bo'yicha eng baland
# ballni oladi, shuning uchun har biridan bittadan vakil yetarli — verdict
# haqiqiy topilmalar ro'yxati bilan bir xil chiqadi.
EXPLICIT_LABEL = sorted(det.EXPLICIT_LABELS)[0]
COVERED_LABEL = sorted(det.SUGGESTIVE_LABELS)[0]
_BOX = Box(x=0, y=0, width=1, height=1)


@dataclass
class Thresholds:
    nsfw: float
    nsfw_confident: float
    nsfl: float
    suggestive: float
    explicit_promote: float

    def as_env(self) -> str:
        return (
            f"THRESHOLD_NSFW={self.nsfw:g}\n"
            f"THRESHOLD_NSFW_CONFIDENT={self.nsfw_confident:g}\n"
            f"THRESHOLD_NSFL={self.nsfl:g}\n"
            f"THRESHOLD_SUGGESTIVE={self.suggestive:g}"
        )


class _SettingsShim:
    """`scoring._settings` o'rniga qo'yiladigan minimal obyekt."""

    def __init__(self, t: Thresholds) -> None:
        self.threshold_nsfw = t.nsfw
        self.threshold_nsfw_confident = t.nsfw_confident
        self.threshold_nsfl = t.nsfl
        self.threshold_suggestive = t.suggestive


def build_detections(explicit_peak: float, covered_peak: float) -> list[Detection]:
    out: list[Detection] = []
    if explicit_peak > 0:
        out.append(Detection(label=EXPLICIT_LABEL, score=float(explicit_peak), box=_BOX))
    if covered_peak > 0:
        out.append(Detection(label=COVERED_LABEL, score=float(covered_peak), box=_BOX))
    out.sort(key=lambda d: d.score, reverse=True)
    return out


def evaluate_all(
    probs: np.ndarray,
    detections: list[list[Detection]],
    t: Thresholds,
) -> np.ndarray:
    """Har bir rasm uchun verdict indeksini qaytaradi (CLASSES tartibida)."""
    original_settings = scoring._settings
    original_promote = scoring.EXPLICIT_PROMOTE_SCORE
    scoring._settings = _SettingsShim(t)
    scoring.EXPLICIT_PROMOTE_SCORE = t.explicit_promote
    try:
        order = {Verdict.SAFE: 0, Verdict.SUGGESTIVE: 1, Verdict.NSFW: 2, Verdict.NSFL: 3}
        out = np.zeros(len(probs), np.int64)
        for i, row in enumerate(probs):
            cs = {"nsfl": float(row[0]), "nsfw": float(row[1]), "sfw": float(row[2])}
            verdict, _c, _s, _r = scoring.evaluate(cs, detections[i])
            out[i] = order[verdict]
        return out
    finally:
        scoring._settings = original_settings
        scoring.EXPLICIT_PROMOTE_SCORE = original_promote


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Bloklash qarori bo'yicha ko'rsatkichlar (nsfw yoki nsfl = bloklash)."""
    pred_block = pred >= 2
    true_block = truth >= 2

    tp = int((pred_block & true_block).sum())
    fp = int((pred_block & ~true_block).sum())
    fn = int((~pred_block & true_block).sum())
    tn = int((~pred_block & ~true_block).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Bolalar platformasi uchun yolg'on ijobiy alohida og'riqli — oddiy
    # rasmlarning necha foizi noto'g'ri bloklanganini alohida ko'rsatamiz.
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset.npz"))
    ap.add_argument("--min-recall", type=float, default=0.90,
                    help="shu recall'dan past variantlar rad etiladi")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    if not args.data.exists():
        print(f"XATO: {args.data} topilmadi. Avval embed.py ni ishga tushiring.")
        return 1

    data = np.load(args.data, allow_pickle=True)
    probs, truth = data["probs"], data["y"]
    classes = [str(c) for c in data["classes"]]
    detections = [
        build_detections(e, c)
        for e, c in zip(data["explicit_peak"], data["covered_peak"], strict=True)
    ]

    counts = np.bincount(truth, minlength=len(classes))
    print("To'plam:", ", ".join(f"{classes[i]}={counts[i]}" for i in range(len(classes)) if counts[i]))
    if (truth >= 2).sum() == 0:
        print("XATO: bloklanishi kerak bo'lgan rasm yo'q (nsfw/nsfl). Kalibrlab bo'lmaydi.")
        return 1
    print()

    from app.config import get_settings

    s = get_settings()
    current = Thresholds(
        nsfw=s.threshold_nsfw,
        nsfw_confident=s.threshold_nsfw_confident,
        nsfl=s.threshold_nsfl,
        suggestive=s.threshold_suggestive,
        explicit_promote=scoring.EXPLICIT_PROMOTE_SCORE,
    )
    base = metrics(evaluate_all(probs, detections, current), truth)
    print("HOZIRGI sozlama:")
    print(f"  nsfw={current.nsfw:g} confident={current.nsfw_confident:g} "
          f"explicit_promote={current.explicit_promote:g} nsfl={current.nsfl:g}")
    print(f"  precision={base['precision']:.3f}  recall={base['recall']:.3f}  "
          f"F1={base['f1']:.3f}  yolg'on-bloklash={base['fpr']:.3f}")
    print(f"  o'tkazib yuborilgan (FN)={base['fn']}  noto'g'ri bloklangan (FP)={base['fp']}")
    print()

    grid = list(itertools.product(
        [50, 55, 60, 65, 70, 75, 80, 85],       # nsfw
        [85, 90, 95, 99],                        # nsfw_confident
        [30, 40, 50, 60, 70],                    # explicit_promote
        [s.threshold_nsfl],                      # nsfl — namuna kam bo'lsa qo'zg'atmaymiz
        [s.threshold_suggestive],
    ))
    print(f"{len(grid)} ta kombinatsiya sinalmoqda ...")

    results: list[tuple[dict, Thresholds]] = []
    for nsfw, conf, promote, nsfl, sugg in grid:
        if conf < nsfw:
            continue  # ma'nosiz: ishonch chegarasi asosiy chegaradan past
        t = Thresholds(nsfw, conf, nsfl, sugg, promote)
        results.append((metrics(evaluate_all(probs, detections, t), truth), t))

    eligible = [r for r in results if r[0]["recall"] >= args.min_recall]
    constrained = bool(eligible)
    if not constrained:
        best_recall = max(r[0]["recall"] for r in results)
        print(f"\n! Hech bir kombinatsiya recall >= {args.min_recall} bermadi "
              f"(eng yaxshisi {best_recall:.3f}).")
        print("  Ma'nosi: chegara emas, MODEL yetarli emas -> train_head.py ga o'ting.")
        eligible = results

    # Shartni qanoatlantirganlar orasidan eng kam yolg'on bloklaydigani.
    eligible.sort(key=lambda r: (-r[0]["f1"], r[0]["fpr"]))

    scope = f"recall >= {args.min_recall}" if constrained else "recall sharti BAJARILMADI"
    print(f"\nEng yaxshi {min(args.top, len(eligible))} ta ({scope}):")
    print(f"  {'nsfw':>5} {'conf':>5} {'promo':>6} | {'prec':>6} {'recall':>7} {'F1':>6} {'yolgFP':>7} | FN  FP")
    for m, t in eligible[:args.top]:
        print(f"  {t.nsfw:5g} {t.nsfw_confident:5g} {t.explicit_promote:6g} | "
              f"{m['precision']:6.3f} {m['recall']:7.3f} {m['f1']:6.3f} {m['fpr']:7.3f} | "
              f"{m['fn']:3d} {m['fp']:3d}")

    best_m, best_t = eligible[0]
    print("\n" + "=" * 62)
    print("TAVSIYA — `.env` ga ko'chiring:")
    print(best_t.as_env())
    print(f"\n  EXPLICIT_PROMOTE_SCORE = {best_t.explicit_promote:g}")
    print("  (bu `.env` da emas — app/services/scoring.py ichida konstanta)")
    delta_f1 = best_m["f1"] - base["f1"]
    print(f"\n  F1: {base['f1']:.3f} -> {best_m['f1']:.3f}  ({delta_f1:+.3f})")
    print(f"  o'tkazib yuborilgan: {base['fn']} -> {best_m['fn']}")
    print(f"  noto'g'ri bloklangan: {base['fp']} -> {best_m['fp']}")
    if delta_f1 < 0.01:
        print("\n  ! Yutuq juda kichik — chegaralar allaqachon yaxshi joyda.")
        print("    Aniqlikni oshirish uchun train_head.py kerak.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
