"""
Eğitilmiş skin_gate.npz'i bir klasördeki görsellerde test eder (yeniden eğitmeden).
    python test_gate.py <klasor>
"""
import os, sys, glob, numpy as np, torch
from predict import load_model, _base_tfm, _img_to_numpy

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
path = sys.argv[1] if len(sys.argv) > 1 else "oodtest"

g = np.load("skin_gate.npz")
W, B, THR = g["w"], float(g["b"]), float(g["threshold"])
model, dev = load_model("skinxai_v6_final.pth", "v6")
tfm = _base_tfm()


@torch.no_grad()
def emb(img):
    t = tfm(image=img)["image"].unsqueeze(0).to(dev)
    f = model.backbone(t)[0].cpu().numpy().astype(np.float32)
    return f / (np.linalg.norm(f) + 1e-8)


files = [f for f in glob.glob(os.path.join(path, "*")) if f.lower().endswith(EXTS)]
print(f"Eşik = {THR:.3f}   ({len(files)} görsel)\n")
for f in files:
    p = 1 / (1 + np.exp(-(emb(_img_to_numpy(f)) @ W + B)))
    karar = "LEZYON" if p >= THR else "RED (OOD)"
    print(f"  {os.path.basename(f):32s} p={p:.3f} -> {karar}")
