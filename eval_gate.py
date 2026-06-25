"""
OOD kapısı (cilt-lezyonu / değil) 
"""
import os, csv, glob, random, numpy as np, torch, torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict import load_model, _base_tfm, _img_to_numpy, _load_skin_gate

random.seed(7)
model, dev = load_model("skinxai_v6_final.pth", "v6")
tfm = _base_tfm()
W, B, THR = _load_skin_gate()


@torch.no_grad()
def prob(img):
    t = tfm(image=img)["image"].unsqueeze(0).to(dev)
    f = model.backbone(t)[0].cpu().numpy().astype(np.float32)
    f /= (np.linalg.norm(f) + 1e-8)
    return float(1 / (1 + np.exp(-(f @ W + B))))


# eğitimde olmayan ISIC lezyonları ──
used = set(os.path.basename(f) for f in glob.glob("dataset/*/*.jpg"))
SRC = os.path.expanduser("~/Downloads/ISIC_2019_Training_Input")
CSV = os.path.expanduser("~/Downloads/ISIC_2019_Training_GroundTruth.csv")
cls = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
pos = []
with open(CSV) as f:
    for row in csv.DictReader(f):
        name = row["image"] + ".jpg"
        if name in used:
            continue
        if any(float(row[c]) == 1.0 for c in cls):
            pos.append(name)
random.shuffle(pos)
pos = pos[:400]
pp = np.array([prob(_img_to_numpy(os.path.join(SRC, n)))
              for n in pos if os.path.exists(os.path.join(SRC, n))])

# HELD-OUT NEGATİF
stl = torchvision.datasets.STL10("_ooddata", split="test", download=False)
idx = random.sample(range(len(stl)), 380)
nn = [prob(np.array(stl[i][0].convert("RGB"))) for i in idx]
for f in glob.glob("oodtest/*"):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            nn.append(prob(_img_to_numpy(f)))
        except Exception:
            pass
nn = np.array(nn)

# ── KARMAŞIKLIK MATRİSİ ──
TP = int((pp >= THR).sum()); FN = int((pp < THR).sum())
TN = int((nn < THR).sum()); FP = int((nn >= THR).sum())
cm = np.array([[TP, FN], [FP, TN]])
labels = ["Cilt Lezyonu", "Lezyon Değil (OOD)"]

acc = (TP + TN) / cm.sum()
sens = TP / (TP + FN)
spec = TN / (TN + FP)

fig, ax = plt.subplots(figsize=(6.2, 5.4))
im = ax.imshow(cm, cmap="GnBu")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Lezyon", "Değil"]); ax.set_yticklabels(labels, rotation=90, va="center")
ax.set_xlabel("Kapının Tahmini", fontsize=12, fontweight="bold")
ax.set_ylabel("Gerçek", fontsize=12, fontweight="bold")
ax.set_title("OOD Kapısı — Karmaşıklık Matrisi (held-out)", fontsize=13, fontweight="bold", pad=14)
mx = cm.max()
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=22, fontweight="bold",
                color="white" if cm[i, j] > mx * 0.5 else "#0B1120")
ax.text(0.5, -0.18, f"Accuracy %{100*acc:.1f}   ·   Sensitivity %{100*sens:.1f}   ·   Specificity %{100*spec:.1f}",
        transform=ax.transAxes, ha="center", fontsize=10.5, color="#334155")
plt.tight_layout()
plt.savefig("assets/ood_confusion_matrix.png", dpi=150, bbox_inches="tight")
print(" assets/ood_confusion_matrix.png")

# ── OLASILIK DAĞILIMI ──
fig, ax = plt.subplots(figsize=(7.2, 4.6))
bins = np.linspace(0, 1, 31)
ax.hist(pp, bins=bins, alpha=0.75, label=f"Gerçek lezyon (n={len(pp)})", color="#0EA5E9")
ax.hist(nn, bins=bins, alpha=0.75, label=f"Lezyon değil / OOD (n={len(nn)})", color="#F59E0B")
ax.axvline(THR, color="#DC2626", ls="--", lw=2, label=f"Eşik = {THR:.2f}")
ax.set_xlabel("Kapının 'lezyon' olasılığı", fontsize=12, fontweight="bold")
ax.set_ylabel("Görüntü sayısı", fontsize=12, fontweight="bold")
ax.set_title("OOD Kapısı — Olasılık Dağılımı (held-out)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("assets/ood_probability_dist.png", dpi=150, bbox_inches="tight")
print("✅ assets/ood_probability_dist.png")

print(f"\nTP={TP} FN={FN} FP={FP} TN={TN}")
print(f"Accuracy %{100*acc:.1f} | Sensitivity %{100*sens:.1f} | Specificity %{100*spec:.1f}")
