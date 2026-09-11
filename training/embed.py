#!/usr/bin/env python
"""1-qadam: belgilangan rasmlarni raqamli to'plamga aylantirish.

Har bir rasm uchun bir marta hisoblanadi va `dataset.npz` ga yoziladi:

  * `X`             — backbone embeddingi (224 o'lchov)
  * `probs`         — hozirgi klassifikator ballari (nsfl, nsfw, sfw)
  * `explicit_peak` — NudeNet'ning eng baland OCHIQ-OYDIN topilmasi
  * `covered_peak`  — eng baland YOPIQ/yarim ochiq topilmasi
  * `y`             — sizning yorlig'ingiz

Shundan keyin `calibrate.py` va `train_head.py` og'ir hisoblashsiz,
sekundlar ichida qayta-qayta ishlaydi.

Kutilgan katalog tuzilishi (yorliq = papka nomi):

    data/safe/*.jpg
    data/suggestive/*.jpg
    data/nsfw/*.jpg
    data/nsfl/*.jpg

Foydalanish:
    .venv/bin/python training/embed.py --data-dir data --out dataset.npz
    .venv/bin/python training/embed.py --urls-csv list.csv --data-dir data  # avval yuklab olish
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import imaging  # noqa: E402
from app.services import detector as det  # noqa: E402
from app.services.detector import BodyPartDetector  # noqa: E402
from training.features import FeatureExtractor, ensure_feature_model  # noqa: E402

CLASSES = ("safe", "suggestive", "nsfw", "nsfl")
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def download(csv_path: Path, data_dir: Path) -> None:
    """`url,label` ustunli CSV bo'yicha rasmlarni papkalarga yuklab oladi.

    Bu offline vosita — servisning "rasm diskka yozilmaydi" qoidasi faqat
    ishlab chiqarish quvuriga tegishli, o'qitish to'plami esa diskda yashaydi.
    """
    import httpx

    with csv_path.open(newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].startswith("#")]

    ok = skipped = failed = 0
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        for i, row in enumerate(rows, 1):
            if len(row) < 2:
                continue
            url, label = row[0].strip(), row[1].strip()
            if label not in CLASSES:
                print(f"  ! noma'lum yorliq {label!r} — o'tkazib yuborildi")
                failed += 1
                continue
            target_dir = data_dir / label
            target_dir.mkdir(parents=True, exist_ok=True)
            # Fayl nomi URL'dan emas, tartib raqamidan — CDN nomlari
            # takrorlanishi va yo'l belgilarini o'z ichiga olishi mumkin.
            stem = f"{i:06d}"
            if any(target_dir.glob(f"{stem}.*")):
                skipped += 1
                continue
            try:
                resp = client.get(url)
                resp.raise_for_status()
                fmt = imaging.sniff_format(resp.content) or "bin"
                (target_dir / f"{stem}.{fmt.lower()}").write_bytes(resp.content)
                ok += 1
            except Exception as exc:  # noqa: BLE001 — hisobot uchun yig'amiz
                print(f"  ! {url[:60]} -> {type(exc).__name__}")
                failed += 1
            if i % 50 == 0:
                print(f"  {i}/{len(rows)} ...")

    print(f"Yuklandi: {ok}, mavjud: {skipped}, xato: {failed}")


def collect(data_dir: Path) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for idx, name in enumerate(CLASSES):
        folder = data_dir / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in SUFFIXES and path.is_file():
                items.append((path, idx))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("dataset.npz"))
    ap.add_argument("--urls-csv", type=Path, help="avval `url,label` CSV bo'yicha yuklab olish")
    ap.add_argument("--threads", type=int, default=8, help="ONNX intra-op oqimlari")
    ap.add_argument("--min-detection-score", type=float, default=25.0)
    args = ap.parse_args()

    if args.urls_csv:
        print(f"CSV bo'yicha yuklab olinmoqda: {args.urls_csv}")
        download(args.urls_csv, args.data_dir)

    items = collect(args.data_dir)
    if not items:
        print(f"XATO: {args.data_dir} ichida rasm topilmadi.")
        print(f"Kutilgan tuzilish: {args.data_dir}/<{'|'.join(CLASSES)}>/*.jpg")
        return 1

    counts = {name: sum(1 for _, y in items if y == i) for i, name in enumerate(CLASSES)}
    print("To'plam:", ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    thin = [k for k, v in counts.items() if 0 < v < 50]
    if thin:
        print(f"  ! juda kam namuna: {', '.join(thin)} (sinfiga kamida 50, yaxshisi 300+)")

    from app.config import get_settings

    settings = get_settings()
    feat_model = ensure_feature_model(
        settings.classifier_path, settings.models_dir / "classifier_with_features.onnx"
    )
    extractor = FeatureExtractor(feat_model, threads=args.threads)
    detector = BodyPartDetector(intra_threads=args.threads)

    n = len(items)
    X = np.zeros((n, 224), np.float32)
    probs = np.zeros((n, 3), np.float32)
    explicit = np.zeros(n, np.float32)
    covered = np.zeros(n, np.float32)
    y = np.zeros(n, np.int64)
    paths: list[str] = []
    bad: list[str] = []

    started = time.perf_counter()
    for i, (path, label) in enumerate(items):
        try:
            img, _info = imaging.decode(path.read_bytes())
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{path}: {type(exc).__name__}")
            paths.append(str(path))
            y[i] = -1  # yaroqsiz — keyin filtrlanadi
            continue

        scores, emb = extractor.run(img)
        raw = detector.detect(imaging.to_bgr_array(img), args.min_detection_score)

        X[i] = emb
        probs[i] = [scores["nsfl"], scores["nsfw"], scores["sfw"]]
        explicit[i] = max(
            (d["score"] for d in raw if d["label"] in det.EXPLICIT_LABELS), default=0.0
        )
        covered[i] = max(
            (d["score"] for d in raw if d["label"] in det.SUGGESTIVE_LABELS), default=0.0
        )
        y[i] = label
        paths.append(str(path))

        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.perf_counter() - started)
            print(f"  {i + 1}/{n}  ({rate:.0f} rasm/s)")

    keep = y >= 0
    if not keep.all():
        print(f"  ! {int((~keep).sum())} ta rasm o'qilmadi:")
        for line in bad[:10]:
            print(f"    {line}")

    np.savez_compressed(
        args.out,
        X=X[keep],
        probs=probs[keep],
        explicit_peak=explicit[keep],
        covered_peak=covered[keep],
        y=y[keep],
        paths=np.array(paths, dtype=object)[keep],
        classes=np.array(CLASSES),
    )
    elapsed = time.perf_counter() - started
    print(f"\nSaqlandi: {args.out}  ({int(keep.sum())} rasm, {elapsed:.1f} s)")
    print(f"Keyingi qadam: .venv/bin/python training/calibrate.py --data {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
