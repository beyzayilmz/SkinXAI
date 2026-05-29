"""
SkinXAI — Ensemble + Cascade Hızlı Test Scripti
================================================
Kullanım modları:

1. CSV ile (etiketli, metrik hesaplar):
   python test_ensemble.py --v3 skinxai_v3_best.pth --v6 skinxai_v6_best.pth \
       --csv test_df.csv --img-col image_path --label-col label

2. Klasör ile (alt klasörler = sınıf isimleri):
   python test_ensemble.py --v3 skinxai_v3_best.pth --v6 skinxai_v6_best.pth \
       --folder test_images/

3. Tek görüntü:
   python test_ensemble.py --v3 skinxai_v3_best.pth --v6 skinxai_v6_best.pth \
       --image lezyon.jpg

Cascade modu (önerilen — expert model ekle):
   python test_ensemble.py --v3 ... --v6 ... --expert skinxai_expert_best.pth \
       --image lezyon.jpg --tta
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score,
)

from predict import (
    predict_image, load_ensemble,
    CLASS_NAMES, CLASS_INFO, HIGH_RISK, RISK_THRESHOLDS,
)

# Renkli terminal çıktısı
try:
    import colorama; colorama.init()
    RED  = "\033[91m"; GRN = "\033[92m"; YLW = "\033[93m"
    BLU  = "\033[94m"; CYN = "\033[96m"; RST = "\033[0m"
    BOLD = "\033[1m"
except ImportError:
    RED = GRN = YLW = BLU = CYN = RST = BOLD = ""


# ──────────────────────────────────────────────
# Yardımcı: tek görüntü tahmin + yazdır
# ──────────────────────────────────────────────
def run_single(image_path, v3_path, v6_path, expert_path=None,
               use_tta=False, true_label=None):
    t0 = time.time()
    result = predict_image(
        image_path,
        ensemble=True,
        v3_path=v3_path,
        v6_path=v6_path,
        expert_path=expert_path,
        use_tta=use_tta,
        cascade=(expert_path is not None),
    )
    elapsed = time.time() - t0

    pred  = result["predicted_class"]
    conf  = result["confidence"]
    risk  = result["is_high_risk"]

    correct = (true_label is None) or (pred == true_label)
    status  = f"{GRN}✓{RST}" if correct else f"{RED}✗{RST}"

    label_str = f"  gerçek={true_label}" if true_label else ""

    risk_str = ""
    if risk:
        risk_str = f"  {RED}⚠ RİSK{RST}"
    elif result.get("warning"):
        risk_str = f"  {YLW}⚠ UYARI{RST}"

    cascade_str = f"  {CYN}[cascade]{RST}" if result.get("cascade_triggered") else ""

    print(f"{status} {Path(image_path).name:<30}  "
          f"tahmin={BOLD}{pred:10s}{RST}  "
          f"güven={conf:.2f}{label_str}{risk_str}{cascade_str}  "
          f"({elapsed*1000:.0f}ms)")

    return pred, true_label, result["all_probs"]


# ──────────────────────────────────────────────
# Metrik raporu
# ──────────────────────────────────────────────
def print_metrics(y_true, y_pred):
    print(f"\n{BOLD}{'='*60}{RST}")
    print(f"{BOLD}SONUÇ RAPORU{RST}")
    print(f"{'='*60}")

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    f1  = f1_score(y_true, y_pred, average="macro",
                   labels=CLASS_NAMES, zero_division=0)
    print(f"Accuracy   : {BOLD}{acc:.1%}{RST}")
    print(f"Macro-F1   : {BOLD}{f1:.4f}{RST}")

    # v3 baseline referansı
    V3 = {"ak":0.877,"bcc":0.875,"bkl":0.667,"df":0.831,
          "melanoma":0.597,"nevus":0.874,"scc":0.540,"vasc":0.962}

    print(f"\n{'Sınıf':12s} {'Recall':>8} {'v3 ref':>8} {'Δ':>7}  {'Durum'}")
    print("-"*50)
    recalls = recall_score(y_true, y_pred, labels=CLASS_NAMES,
                           average=None, zero_division=0)
    for cls, rc in zip(CLASS_NAMES, recalls):
        v3r   = V3.get(cls, 0)
        delta = rc - v3r
        arrow = f"{GRN}↑{RST}" if delta > 0.02 else (f"{RED}↓{RST}" if delta < -0.02 else "→")
        flag  = f" {RED}⚠{RST}" if cls in HIGH_RISK else ""
        print(f"{cls:12s} {rc:8.1%} {v3r:8.1%} {delta:+7.1%}  {arrow}{flag}")

    print(f"\n{BOLD}Detaylı Rapor:{RST}")
    labels_present = sorted(set(y_true) | set(y_pred),
                            key=lambda x: CLASS_NAMES.index(x) if x in CLASS_NAMES else 99)
    print(classification_report(y_true, y_pred, labels=labels_present,
                                 zero_division=0, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    print(f"{BOLD}Confusion Matrix (satır=gerçek, sütun=tahmin):{RST}")
    header = "         " + "".join(f"{c[:4]:>6}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(
            (f"{GRN}{v:6d}{RST}" if i == j else
             (f"{RED}{v:6d}{RST}" if v > 0 else f"{'0':>6}"))
            for j, v in enumerate(row)
        )
        print(f"{CLASS_NAMES[i]:9s}{row_str}")


# ──────────────────────────────────────────────
# MOD 1: CSV
# ──────────────────────────────────────────────
def run_csv(args):
    df = pd.read_csv(args.csv)
    img_col   = args.img_col   or "image_path"
    label_col = args.label_col or "label"

    if img_col not in df.columns:
        print(f"{RED}Hata: '{img_col}' kolonu CSV'de yok.{RST}")
        print(f"Mevcut kolonlar: {list(df.columns)}"); sys.exit(1)

    has_labels = label_col in df.columns
    if not has_labels:
        print(f"{YLW}Uyarı: '{label_col}' kolonu yok → metrik hesaplanamaz.{RST}")

    load_ensemble(args.v3, args.v6, args.expert)
    cascade_str = " + Expert Cascade" if args.expert else ""
    print(f"\n{BOLD}Test ediliyor: {len(df)} görüntü  |  TTA: {args.tta}{cascade_str}{RST}\n")

    y_true, y_pred = [], []
    errors = 0

    for _, row in df.iterrows():
        img_path = row[img_col]
        true_lbl = row[label_col] if has_labels else None

        if not os.path.exists(img_path):
            print(f"{YLW}⚠ Bulunamadı: {img_path}{RST}")
            errors += 1; continue

        try:
            pred, lbl, _ = run_single(
                img_path, args.v3, args.v6, args.expert,
                use_tta=args.tta, true_label=true_lbl,
            )
            if has_labels:
                y_true.append(lbl); y_pred.append(pred)
        except Exception as e:
            print(f"{RED}Hata ({img_path}): {e}{RST}")
            errors += 1

    if errors:
        print(f"\n{YLW}{errors} görüntü atlandı.{RST}")
    if has_labels and y_true:
        print_metrics(y_true, y_pred)


# ──────────────────────────────────────────────
# MOD 2: Klasör
# ──────────────────────────────────────────────
def run_folder(args):
    folder = Path(args.folder)
    if not folder.exists():
        print(f"{RED}Klasör bulunamadı: {folder}{RST}"); sys.exit(1)

    subdirs    = [d for d in folder.iterdir() if d.is_dir()]
    has_labels = len(subdirs) > 0 and all(d.name in CLASS_NAMES for d in subdirs)
    exts       = {".jpg", ".jpeg", ".png", ".bmp"}

    if has_labels:
        rows = []
        for sd in subdirs:
            for f in sd.iterdir():
                if f.suffix.lower() in exts:
                    rows.append({"image_path": str(f), "label": sd.name})
        df = pd.DataFrame(rows)
        print(f"{GRN}Etiketli klasör: {len(df)} görüntü, {len(subdirs)} sınıf{RST}")
    else:
        imgs = [f for f in folder.rglob("*") if f.suffix.lower() in exts]
        df   = pd.DataFrame({"image_path": [str(f) for f in imgs]})
        print(f"{YLW}Etiketsiz klasör: {len(df)} görüntü (metrik hesaplanamaz){RST}")

    load_ensemble(args.v3, args.v6, args.expert)
    cascade_str = " + Expert Cascade" if args.expert else ""
    print(f"\n{BOLD}Test ediliyor: {len(df)} görüntü  |  TTA: {args.tta}{cascade_str}{RST}\n")

    y_true, y_pred = [], []
    for _, row in df.iterrows():
        true_lbl = row.get("label")
        try:
            pred, lbl, _ = run_single(
                row["image_path"], args.v3, args.v6, args.expert,
                use_tta=args.tta, true_label=true_lbl,
            )
            if true_lbl:
                y_true.append(lbl); y_pred.append(pred)
        except Exception as e:
            print(f"{RED}Hata: {e}{RST}")

    if y_true:
        print_metrics(y_true, y_pred)


# ──────────────────────────────────────────────
# MOD 3: Tek görüntü
# ──────────────────────────────────────────────
def run_image(args):
    load_ensemble(args.v3, args.v6, args.expert)
    print()

    result = predict_image(
        args.image,
        ensemble=True,
        v3_path=args.v3,
        v6_path=args.v6,
        expert_path=args.expert,
        use_tta=args.tta,
        cascade=(args.expert is not None),
    )

    print(f"{BOLD}{'='*55}{RST}")
    print(f"{BOLD}TAHMİN SONUCU  [{result['mode'].upper()}]{RST}")
    print(f"{'='*55}")
    pred = result["predicted_class"]
    col  = RED if result["is_high_risk"] else GRN
    print(f"Sınıf    : {col}{BOLD}{pred}{RST}  —  {result['full_name']}")
    print(f"Güven    : {result['confidence_pct']}")
    if result.get("cascade_triggered"):
        print(f"Cascade  : {CYN}Tetiklendi → ak/bkl/scc expert modelle rafine edildi{RST}")
    if result["warning"]:
        print(f"\n{result['warning']}")

    print(f"\n{BOLD}Tüm Olasılıklar (Stage 1):{RST}")
    for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar  = "█" * int(prob * 35)
        flag = f" {RED}⚠{RST}" if cls in HIGH_RISK and prob >= RISK_THRESHOLDS.get(cls, 1.0) else ""
        hi   = BOLD if cls == pred else ""
        print(f"  {hi}{cls:10s}{RST}: {prob:.3f}  {bar}{flag}")

    if result.get("expert_probs"):
        print(f"\n{BOLD}Expert Model (ak/bkl/scc):{RST}")
        for cls, prob in sorted(result["expert_probs"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 35)
            hi  = BOLD if cls == pred else ""
            print(f"  {hi}{cls:10s}{RST}: {prob:.3f}  {bar}")


# ──────────────────────────────────────────────
# ARGPARSE
# ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="SkinXAI ensemble + cascade test scripti")
    p.add_argument("--v3",     required=True,  help="v3 model .pth yolu")
    p.add_argument("--v6",     required=True,  help="v6 model .pth yolu")
    p.add_argument("--expert", default=None,   help="Expert cascade .pth yolu (isteğe bağlı)")
    p.add_argument("--tta",    action="store_true", help="TTA kullan")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv",    help="Etiketli CSV dosyası")
    g.add_argument("--folder", help="Test görüntüleri klasörü")
    g.add_argument("--image",  help="Tek görüntü dosyası")

    p.add_argument("--img-col",   default="image_path")
    p.add_argument("--label-col", default="label")

    args = p.parse_args()

    for path, name in [(args.v3, "v3"), (args.v6, "v6")]:
        if not os.path.exists(path):
            print(f"{RED}Hata: {name} modeli bulunamadı → {path}{RST}")
            sys.exit(1)

    if args.expert and not os.path.exists(args.expert):
        print(f"{RED}Hata: expert modeli bulunamadı → {args.expert}{RST}")
        sys.exit(1)

    if args.csv:      run_csv(args)
    elif args.folder: run_folder(args)
    else:             run_image(args)


if __name__ == "__main__":
    main()
