#!/usr/bin/env python
"""Daraja 1: backbone'ni qotirib, faqat oxirgi qatlamni qayta o'qitish.

Backbone (SwiftFormer-S) rasmni 224 o'lchovli vektorga aylantiradi — bu qism
tegilmaydi. Uning ustidagi mayda chiziqli qatlam esa sizning ma'lumotingizda
qaytadan o'qitiladi. PyTorch kerak emas, o'qitish sekundlar ichida tugaydi,
shuning uchun gipeparametrlarni erkin sinab ko'rish mumkin.

Chiqish — `models/custom_head.npz`. Uni ishlatish uchun `.env` ga:

    CUSTOM_HEAD_FILE=custom_head.npz

Sinflar tartibi asl modeldagidek qoladi (nsfl, nsfw, sfw), shu sababli
`scoring.py`, `.env` chegaralari va API javob formati o'zgarmaydi.

Foydalanish:
    .venv/bin/python training/train_head.py --data dataset.npz
    .venv/bin/python training/train_head.py --data dataset.npz --suggestive-as nsfw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.classifier import LABELS  # noqa: E402  (nsfl, nsfw, sfw)

DATA_CLASSES = ("safe", "suggestive", "nsfw", "nsfl")


def map_labels(y: np.ndarray, suggestive_as: str) -> np.ndarray:
    """4 sinfli yorliqlarni modelning 3 sinfiga o'tkazadi.

    `suggestive` — klassifikatorning sinfi emas: u detektor topilmalaridan
    hosil bo'ladi. Shuning uchun uni `sfw` ga qo'shish standart yo'l —
    klassifikator "ochiq teri = nsfw" deb o'rganishining oldini oladi
    (aynan shu xato ko'kragi ochiq bolakayga 78% bergan edi).
    """
    index = {name: i for i, name in enumerate(LABELS)}
    mapping = {
        DATA_CLASSES.index("safe"): index["sfw"],
        DATA_CLASSES.index("suggestive"): index[suggestive_as],
        DATA_CLASSES.index("nsfw"): index["nsfw"],
        DATA_CLASSES.index("nsfl"): index["nsfl"],
    }
    return np.array([mapping[int(v)] for v in y], np.int64)


def stratified_split(y: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * val_frac))) if len(idx) > 1 else 0
        val_idx.append(idx[:cut])
        train_idx.append(idx[cut:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    steps: int,
    lr: float,
    l2: float,
    class_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Adam bilan multinomial logistic regression."""
    n, d = X.shape
    W = np.zeros((d, n_classes), np.float32)
    b = np.zeros(n_classes, np.float32)
    Y = np.eye(n_classes, dtype=np.float32)[y]
    w_i = class_weight[y][:, None]  # namuna og'irliklari
    w_sum = w_i.sum()

    mW = vW = np.zeros_like(W)
    mb = vb = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, steps + 1):
        P = softmax(X @ W + b)
        G = (P - Y) * w_i / w_sum
        gW = X.T @ G + l2 * W
        gb = G.sum(axis=0)

        mW = beta1 * mW + (1 - beta1) * gW
        vW = beta2 * vW + (1 - beta2) * gW**2
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * gb**2
        W -= lr * (mW / (1 - beta1**step)) / (np.sqrt(vW / (1 - beta2**step)) + eps)
        b -= lr * (mb / (1 - beta1**step)) / (np.sqrt(vb / (1 - beta2**step)) + eps)

    return W, b


def report(name: str, probs: np.ndarray, y: np.ndarray) -> float:
    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean())
    print(f"\n  {name}  (aniqlik {acc * 100:.1f}%)")
    print(f"    {'sinf':>6} {'precision':>10} {'recall':>8} {'n':>6}")
    for i, label in enumerate(LABELS):
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        fn = int(((pred != i) & (y == i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print(f"    {label:>6} {p:10.3f} {r:8.3f} {int((y == i).sum()):6d}")

    # Bloklash qarori: nsfw yoki nsfl.
    pred_block, true_block = pred != LABELS.index("sfw"), y != LABELS.index("sfw")
    tp = int((pred_block & true_block).sum())
    fp = int((pred_block & ~true_block).sum())
    fn = int((~pred_block & true_block).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"    bloklash: precision={prec:.3f} recall={rec:.3f} F1={f1:.3f} (FP={fp}, FN={fn})")
    return f1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset.npz"))
    ap.add_argument("--out", type=Path, default=Path("models/custom_head.npz"))
    ap.add_argument("--suggestive-as", choices=["sfw", "nsfw"], default="sfw")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-class-weight", action="store_true",
                    help="sinflar nomutanosibligini tekislamaslik")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"XATO: {args.data} topilmadi. Avval embed.py ni ishga tushiring.")
        return 1

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = map_labels(data["y"], args.suggestive_as)
    baseline_probs = data["probs"].astype(np.float32) / 100.0  # (nsfl, nsfw, sfw)

    counts = np.bincount(y, minlength=len(LABELS))
    print("Sinflar (model tartibida):",
          ", ".join(f"{LABELS[i]}={counts[i]}" for i in range(len(LABELS))))
    if (counts > 0).sum() < 2:
        print("XATO: kamida ikkita sinf kerak.")
        return 1
    thin = [LABELS[i] for i, c in enumerate(counts) if 0 < c < 30]
    if thin:
        print(f"  ! juda kam namuna: {', '.join(thin)} — natijaga ishonmang")

    train_idx, val_idx = stratified_split(y, args.val_frac, args.seed)
    print(f"O'qitish: {len(train_idx)}, tekshiruv: {len(val_idx)}")

    if args.no_class_weight:
        class_weight = np.ones(len(LABELS), np.float32)
    else:
        # Kam uchraydigan sinfga kattaroq og'irlik — aks holda model
        # ko'pchilik sinfni bashorat qilib ham "yuqori aniqlik" ko'rsatadi.
        safe_counts = np.maximum(np.bincount(y[train_idx], minlength=len(LABELS)), 1)
        class_weight = (len(train_idx) / (len(LABELS) * safe_counts)).astype(np.float32)

    W, b = train(
        X[train_idx], y[train_idx], len(LABELS),
        steps=args.steps, lr=args.lr, l2=args.l2, class_weight=class_weight,
    )

    print("\n" + "=" * 58)
    print("TEKSHIRUV TO'PLAMIDA (model bu rasmlarni ko'rmagan)")
    print("=" * 58)
    base_f1 = report("HOZIRGI bosh (asl model)", baseline_probs[val_idx], y[val_idx])
    new_f1 = report("YANGI bosh (siz o'qitgan)", softmax(X[val_idx] @ W + b), y[val_idx])

    print("\n" + "=" * 58)
    print(f"Bloklash F1: {base_f1:.3f} -> {new_f1:.3f}  ({new_f1 - base_f1:+.3f})")
    if new_f1 <= base_f1:
        print("! Yangi bosh yaxshiroq emas. Sabablari odatda:")
        print("  - ma'lumot juda kam (sinfiga 300+ kerak)")
        print("  - yorliqlar nomuvofiq")
        print("  - to'plam ishlab chiqarish trafikidan farq qiladi")
        print("  Saqlanadi, lekin `.env` ga ulashdan oldin o'ylab ko'ring.")
    print("=" * 58)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        W=W, b=b,
        classes=np.array(LABELS),
        suggestive_as=np.array(args.suggestive_as),
        val_f1=np.array(new_f1),
        baseline_f1=np.array(base_f1),
        n_train=np.array(len(train_idx)),
    )
    print(f"\nSaqlandi: {args.out}")
    print("Ulash uchun `.env` ga qo'shing va servisni qayta ishga tushiring:")
    print(f"  CUSTOM_HEAD_FILE={args.out.name}")
    print("  systemctl restart nsfw-api")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
