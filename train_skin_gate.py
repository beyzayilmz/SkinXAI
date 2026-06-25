
import os, sys, glob, numpy as np, torch
import torch.nn.functional as F
from predict import load_model, _base_tfm, _img_to_numpy

DATASET = sys.argv[1] if len(sys.argv) > 1 else "dataset"
OODTEST = sys.argv[2] if len(sys.argv) > 2 else "oodtest"
N_POS_PER_CLASS = 400      # sınıf başına en fazla pozitif
N_NEG           = 3000     # negatif (STL-10) sayısı
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

model, dev = load_model("skinxai_v6_final.pth", "v6")
tfm = _base_tfm()


@torch.no_grad()
def embed_batch(imgs_rgb):
    """Bir liste RGB görselin L2-normalize embedding'lerini döner (batch)."""
    ts = torch.stack([tfm(image=im)["image"] for im in imgs_rgb]).to(dev)
    f = model.backbone(ts).cpu().numpy().astype(np.float32)
    f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    return f


def embed_paths(paths, label):
    out = []
    B = 32
    for i in range(0, len(paths), B):
        chunk = []
        for p in paths[i:i+B]:
            try:
                chunk.append(_img_to_numpy(p))
            except Exception:
                pass
        if chunk:
            out.append(embed_batch(chunk))
        print(f"  [{label}] {min(i+B,len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(out) if out else np.zeros((0, 1280), np.float32)


# POZİTİFLER 
print("Pozitifler (lezyonlar) toplanıyor...")
pos_paths = []
subdirs = [d for d in sorted(glob.glob(os.path.join(DATASET, "*"))) if os.path.isdir(d)]
if not subdirs:   # düz klasör
    subdirs = [DATASET]
rng = np.random.RandomState(42)
for d in subdirs:
    files = [f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(EXTS)]
    rng.shuffle(files)
    files = files[:N_POS_PER_CLASS]
    pos_paths += files
    print(f"  {os.path.basename(d):12s}: {len(files)}")
print(f"Toplam pozitif: {len(pos_paths)}")
Xpos = embed_paths(pos_paths, "pos")

#  NEGATİFLER 
print("Negatifler toplanıyor (STL-10)...")
import torchvision
stl = torchvision.datasets.STL10("_ooddata", split="unlabeled", download=False)
idx = rng.choice(len(stl), size=min(N_NEG, len(stl)), replace=False)
neg_imgs = [np.array(stl[i][0].convert("RGB")) for i in idx]
Xneg = []
for i in range(0, len(neg_imgs), 32):
    Xneg.append(embed_batch(neg_imgs[i:i+32]))
    print(f"  [neg] {min(i+32,len(neg_imgs))}/{len(neg_imgs)}", end="\r")
print()
Xneg = np.concatenate(Xneg)
# repo grafikleri de negatif
extra_neg = [f for f in glob.glob("negatives/*") if f.lower().endswith(EXTS)]
if extra_neg:
    Xneg = np.concatenate([Xneg, embed_paths(extra_neg, "neg-extra")])

# YÜKSEK ÇÖZÜNÜRLÜKLÜ negatifler (kısa-yol/çözünürlük sorununu engellemek için)
hr_neg = [f for f in glob.glob("_ooddata/highres_neg/*") if f.lower().endswith(EXTS)]
if hr_neg:
    Xneg = np.concatenate([Xneg, embed_paths(hr_neg, "neg-hires")])
    print(f"  {len(hr_neg)} yüksek-çözünürlüklü negatif eklendi")

# LFW yüzleri negatif 
try:
    lfw = torchvision.datasets.LFWPeople("_ooddata", split="train",
                                         image_set="funneled", download=False)
    n_face = min(1000, len(lfw))
    fidx = rng.choice(len(lfw), size=n_face, replace=False)
    face_imgs = [np.array(lfw[i][0].convert("RGB")) for i in fidx]
    Xface = []
    for i in range(0, len(face_imgs), 32):
        Xface.append(embed_batch(face_imgs[i:i+32]))
        print(f"  [yüz] {min(i+32,len(face_imgs))}/{len(face_imgs)}", end="\r")
    print()
    Xneg = np.concatenate([Xneg, np.concatenate(Xface)])
    print(f"  {n_face} yüz negatife eklendi")
except Exception as e:
    print(f"  (LFW yüz eklenemedi: {e})")

print(f"Toplam negatif: {len(Xneg)}")

# EĞİTİM/VALİDASYON BÖLÜMÜ 
def split(X, frac=0.8):
    n = len(X); k = int(n*frac); p = rng.permutation(n)
    return X[p[:k]], X[p[k:]]
Xtr_p, Xva_p = split(Xpos)
Xtr_n, Xva_n = split(Xneg)

Xtr = np.concatenate([Xtr_p, Xtr_n])
ytr = np.concatenate([np.ones(len(Xtr_p)), np.zeros(len(Xtr_n))])
Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.float32)

# LOJİSTİK REGRESYON 
w = torch.zeros(1280, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-3)
# sınıf dengesi: pozitif ağırlığı = neg/pos
pos_w = torch.tensor([len(Xtr_n)/max(1, len(Xtr_p))], dtype=torch.float32)
for it in range(1500):
    opt.zero_grad()
    logit = Xtr_t @ w + b
    loss = F.binary_cross_entropy_with_logits(logit, ytr_t, pos_weight=pos_w)
    loss.backward(); opt.step()
print(f"Eğitim bitti. Son kayıp: {loss.item():.4f}")

W = w.detach().numpy(); B = float(b.detach().numpy()[0])
def prob(X): return 1/(1+np.exp(-(X@W + B)))

# EŞİK KALİBRASYONU: LEZYON GÜVENLİĞİ 
# Validasyon pozitiflerinin %99'u geçecek şekilde eşik seç (yalancı negatif↓)
pva = prob(Xva_p)
THR = float(np.percentile(pva, 1))     # pozitiflerin %1'i bunun altında
THR = max(0.05, min(THR, 0.5))         # makul aralığa sıkıştır

pva_n = prob(Xva_n)
print(f"\n── VALİDASYON ──")
print(f"Eşik (lezyon-güvenli): {THR:.3f}")
print(f"Gerçek lezyon GEÇİŞ oranı: %{100*(pva>=THR).mean():.1f}  (yanlış red %{100*(pva<THR).mean():.1f})")
print(f"Negatif RED oranı:        %{100*(pva_n<THR).mean():.1f}  (kaçan %{100*(pva_n>=THR).mean():.1f})")

# samples ve oodtest üzerinde gerçek test 
def test_dir(path, beklenen):
    files = [f for f in glob.glob(os.path.join(path, "*")) if f.lower().endswith(EXTS)]
    if not files: return
    X = embed_paths(files, os.path.basename(path))
    ps = prob(X)
    for f, p in sorted(zip(files, ps), key=lambda x: x[1]):
        karar = "LEZYON" if p >= THR else "RED(OOD)"
        print(f"  {os.path.basename(f):28s} p={p:.3f} -> {karar}")

print("\n── samples/ (hepsi LEZYON çıkmalı) ──")
test_dir("samples", "lezyon")
if os.path.isdir(OODTEST):
    print(f"\n── {OODTEST}/ (hepsi RED çıkmalı) ──")
    test_dir(OODTEST, "ood")

# ── KAYDET ───────────────────────────────────────────────────
np.savez("skin_gate.npz", w=W, b=B, threshold=THR)
print(f"\n skin_gate.npz kaydedildi (w[1280], b, threshold={THR:.3f})")
