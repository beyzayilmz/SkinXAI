"""
SkinXAI — Inference Modülü  (Ensemble v3 + v6 + Expert Cascade)
================================================================
Tek model:
    from predict import predict_image
    result = predict_image("lezyon.jpg", model_path="skinxai_v3_best.pth")

Ensemble (v3 + v6):
    result = predict_image(
        "lezyon.jpg",
        ensemble=True,
        v3_path="skinxai_v3_best.pth",
        v6_path="skinxai_v6_best.pth",
    )

Ensemble + Expert Cascade (önerilen):
    result = predict_image(
        "lezyon.jpg",
        ensemble=True,
        v3_path="skinxai_v3_best.pth",
        v6_path="skinxai_v6_best.pth",
        cascade=True,
        expert_path="skinxai_expert_best.pth",
        use_tta=True,
    )

    print(result["predicted_class"])   # "scc"
    print(result["confidence"])        # 0.81
    print(result["mode"])              # "cascade"
    print(result["warning"])           # uyarı mesajı veya None
"""

import os
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from PIL import Image


# ============================================================
# SINIF İSİMLERİ ve AÇIKLAMALAR
# ============================================================
CLASS_NAMES = ["ak", "bcc", "bkl", "df", "melanoma", "nevus", "scc", "vasc"]

CLASS_INFO = {
    "ak":       "Aktinik Keratoz (Actinic Keratosis)",
    "bcc":      "Bazal Hücreli Karsinom (Basal Cell Carcinoma)",
    "bkl":      "Benign Keratoz (Benign Keratosis-like Lesion)",
    "df":       "Dermatofibrom (Dermatofibroma)",
    "melanoma": "Melanoma",
    "nevus":    "Nevus (Ben)",
    "scc":      "Skuamöz Hücreli Karsinom (Squamous Cell Carcinoma)",
    "vasc":     "Vasküler Lezyon (Vascular Lesion)",
}

HIGH_RISK = {"melanoma", "bcc", "scc", "ak"}

# Expert model sınıfları (cascade 2. aşama)
EXPERT_CLASS_NAMES = ["ak", "bkl", "scc"]

# Stage 1 bu sınıflardan birini tahmin ettiğinde cascade devreye girer
CASCADE_TRIGGER = {"ak", "bkl", "scc"}


# ============================================================
# ENSEMBLE AĞIRLIKLARI — PER-CLASS (v3 + v6)
# ============================================================
# v3: melanoma ve ak'ta güçlü  (mel=59.7%, ak=87.7%)
# v6: scc, bkl, bcc, nevus'ta güçlü  (scc=58.7%, bkl=87.3%, bcc=96.4%)
# Her sınıf için ayrı (v3_ağırlık, v6_ağırlık) — toplam 1.0
CLASS_ENSEMBLE_WEIGHTS = {
    "melanoma": (0.70, 0.30),   # v3 ağır — v3 melanomada daha iyi
    "ak":       (0.80, 0.20),   # v3 ağır — v3 ak'ta çok daha iyi
    "bcc":      (0.10, 0.90),   # v6 ağır — v6 bcc=96.4%
    "bkl":      (0.10, 0.90),   # v6 ağır — v6 bkl=87.3%
    "scc":      (0.30, 0.70),   # v6 ağır — cascade zaten düzeltiyor
    "df":       (0.40, 0.60),
    "nevus":    (0.30, 0.70),
    "vasc":     (0.40, 0.60),
}


# ============================================================
# THRESHOLD AYARLARI — False Negative Azaltma
# ============================================================
# Mantık: "model scc ihtimali 0.10 diyor" → argmax benign bile olsa uyarı ver.
# Kaçırmanın maliyeti yanlış alarmdan çok daha yüksek.
RISK_THRESHOLDS = {
    "melanoma": 0.10,   # agresif — melanoma hiç kaçmamalı
    "bcc":      0.42,
    "scc":      0.10,   # expert model SCC'yi zaten %87.8 yakalar
    "ak":       0.18,
}


# ============================================================
# MODEL MİMARİLERİ
# ============================================================
class SkinModelV3(nn.Module):
    """v3 mimarisi: Dropout → Linear(d, 8)"""
    def __init__(self, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=False, num_classes=0, global_pool="avg")
        d = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


class SkinModelV6(nn.Module):
    """v6 mimarisi: BN → Dropout(0.4) → Linear(d,512) → SiLU → Dropout(0.2) → Linear(512,8)"""
    def __init__(self, num_classes: int = 8, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=False, num_classes=0, global_pool="avg")
        d = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Dropout(dropout),
            nn.Linear(d, 512),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class ExpertModel(nn.Module):
    """Expert cascade: BN → Dropout(0.4) → Linear(d,256) → SiLU → Dropout(0.2) → Linear(256,3)"""
    def __init__(self, num_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=False, num_classes=0, global_pool="avg")
        d = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Dropout(dropout),
            nn.Linear(d, 256),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ============================================================
# MODEL CACHE — bir kez yükle, hafızada tut
# ============================================================
_model_cache: dict = {}
_device = None


def _resolve_device(device_str: str = None) -> torch.device:
    global _device
    if _device is None:
        _device = torch.device(device_str) if device_str else \
                  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model(model_path: str, version: str = "v3", device: str = None):
    """
    Modeli yükler, cache'e alır. İkinci çağrıda cache'den döner.

    Args:
        model_path : .pth dosya yolu
        version    : "v3" | "v6" | "expert"
        device     : "cuda" | "cpu" | None (otomatik)

    Returns:
        (model, device) tuple
    """
    if version in _model_cache:
        return _model_cache[version]

    dev = _resolve_device(device)

    if version == "v3":
        model = SkinModelV3(num_classes=8).to(dev)
    elif version == "v6":
        model = SkinModelV6(num_classes=8).to(dev)
    elif version == "expert":
        model = ExpertModel(num_classes=3).to(dev)
    else:
        raise ValueError(f"Bilinmeyen versiyon: {version}")

    ckpt = torch.load(model_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _model_cache[version] = (model, dev)
    print(f"[SkinXAI] {version} yüklendi ✅  |  {model_path}  |  Cihaz: {dev}")
    return model, dev


def load_ensemble(v3_path: str, v6_path: str, expert_path: str = None, device: str = None):
    """Tüm modelleri önceden yükler."""
    load_model(v3_path, "v3", device)
    load_model(v6_path, "v6", device)
    if expert_path:
        load_model(expert_path, "expert", device)
        print("[SkinXAI] Cascade hazır ✅  v3 + v6 + expert")
    else:
        print("[SkinXAI] Ensemble hazır ✅  v3 + v6")


# ============================================================
# TRANSFORM
# ============================================================
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD  = [0.229, 0.224, 0.225]

def _base_tfm():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
        ToTensorV2(),
    ])

_TTA_TRANSFORMS = [
    A.Compose([A.Resize(224,224), A.Normalize(mean=_NORM_MEAN,std=_NORM_STD), ToTensorV2()]),
    A.Compose([A.Resize(224,224), A.HorizontalFlip(p=1), A.Normalize(mean=_NORM_MEAN,std=_NORM_STD), ToTensorV2()]),
    A.Compose([A.Resize(224,224), A.VerticalFlip(p=1),   A.Normalize(mean=_NORM_MEAN,std=_NORM_STD), ToTensorV2()]),
    A.Compose([A.Resize(256,256), A.CenterCrop(224,224), A.Normalize(mean=_NORM_MEAN,std=_NORM_STD), ToTensorV2()]),
    A.Compose([A.Resize(224,224), A.Rotate(limit=15,p=1),A.Normalize(mean=_NORM_MEAN,std=_NORM_STD), ToTensorV2()]),
]


# ============================================================
# TAHMİN YARDIMCI FONKSİYONLARI
# ============================================================
@torch.no_grad()
def _infer_single(model, device, img_rgb: np.ndarray, tfm) -> np.ndarray:
    """Tek geçişte softmax olasılık vektörü döner."""
    tensor = tfm(image=img_rgb)["image"].unsqueeze(0).to(device)
    return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()


@torch.no_grad()
def _infer_tta(model, device, img_rgb: np.ndarray) -> np.ndarray:
    """5 TTA geçişinin ortalamasını döner."""
    return np.mean([
        torch.softmax(model(tfm(image=img_rgb)["image"].unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
        for tfm in _TTA_TRANSFORMS
    ], axis=0)


def _img_to_numpy(image_input) -> np.ndarray:
    """Çeşitli girdi formatlarını H×W×3 RGB numpy array'e çevirir."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {image_input}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        return image_input[:, :, :3] if image_input.shape[-1] == 4 else image_input
    else:
        raise TypeError(f"Desteklenmeyen girdi tipi: {type(image_input)}")


def _per_class_ensemble(p3: np.ndarray, p6: np.ndarray) -> np.ndarray:
    """
    Her sınıf için ayrı ağırlıkla v3 + v6 karıştırır.
    CLASS_ENSEMBLE_WEIGHTS kullanır.
    """
    result = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for i, cls in enumerate(CLASS_NAMES):
        w3, w6 = CLASS_ENSEMBLE_WEIGHTS[cls]
        result[i] = w3 * p3[i] + w6 * p6[i]
    # Yeniden normalize et (ağırlıklar sınıfa göre değiştiği için toplam != 1 olabilir)
    result /= result.sum()
    return result


# ============================================================
# ANA TAHMİN FONKSİYONU
# ============================================================
def predict_image(
    image_input,
    model_path: str = None,
    version: str = "v3",
    v3_path: str = None,
    v6_path: str = None,
    expert_path: str = None,
    use_tta: bool = False,
    sensitive_mode: bool = False,
    ensemble: bool = False,
    cascade: bool = False,
    weights: tuple = None,
) -> dict:
    """
    Cilt lezyon görüntüsünü sınıflandırır.

    Parametreler
    ------------
    image_input   : Dosya yolu (str) | numpy array (H,W,3 RGB) | PIL Image
    model_path    : Tek model .pth yolu (ensemble=False durumunda)
    v3_path       : v3 .pth yolu
    v6_path       : v6 .pth yolu
    expert_path   : Expert cascade .pth yolu (cascade=True için gerekli)
    use_tta       : Test Time Augmentation (5 geçiş, ~5× yavaş, daha doğru)
    sensitive_mode: True → threshold'u geçen riskli sınıfı tahmin olarak döner (FN↓)
    ensemble      : True → v3+v6 per-class ağırlıklı ensemble kullan
    cascade       : True → ak/bkl/scc tahminlerini expert modelle rafine et
    weights       : (v3_ağırlık, v6_ağırlık) — None → CLASS_ENSEMBLE_WEIGHTS kullan

    Döndürür
    --------
    dict:
        predicted_class  : "scc"
        full_name        : "Skuamöz Hücreli Karsinom..."
        confidence       : 0.81
        confidence_pct   : "81.0%"
        is_high_risk     : True
        is_uncertain     : False
        risk_warnings    : [("melanoma", 0.12), ...]
        all_probs        : {"ak": 0.05, "bcc": 0.01, ...}  (8 sınıf, Stage 1)
        expert_probs     : {"ak": 0.08, "bkl": 0.11, "scc": 0.81} veya None
        warning          : "⚠️ ..." veya None
        mode             : "cascade" | "ensemble" | "single"
        cascade_triggered: True/False
    """
    img = _img_to_numpy(image_input)
    tfm = _base_tfm()

    # ── Stage 1: Ana ensemble olasılık vektörü ────────────────
    if ensemble:
        if v3_path is None:
            v3_path = os.environ.get("V3_MODEL_PATH", "skinxai_v3_best.pth")
        if v6_path is None:
            v6_path = os.environ.get("V6_MODEL_PATH", "skinxai_v6_best.pth")

        if "v3" not in _model_cache:
            load_model(v3_path, "v3")
        if "v6" not in _model_cache:
            load_model(v6_path, "v6")

        model_v3, device = _model_cache["v3"]
        model_v6, _      = _model_cache["v6"]

        if use_tta:
            p3 = _infer_tta(model_v3, device, img)
            p6 = _infer_tta(model_v6, device, img)
        else:
            p3 = _infer_single(model_v3, device, img, tfm)
            p6 = _infer_single(model_v6, device, img, tfm)

        # Per-class ağırlıklı birleştirme (sabit weights override'ı varsa düz karıştır)
        if weights is not None:
            w3, w6 = weights
            probs = w3 * p3 + w6 * p6
        else:
            probs = _per_class_ensemble(p3, p6)

        mode = "ensemble"

    else:
        # Tek model — geriye dönük uyumlu
        if model_path is None:
            model_path = os.environ.get("MODEL_PATH", "skinxai_v3_best.pth")
        model, device = load_model(model_path, version)
        probs = _infer_tta(model, device, img) if use_tta else _infer_single(model, device, img, tfm)
        mode  = "single"

    # Stage 1 argmax
    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]

    # ── Stage 2: Expert Cascade ───────────────────────────────
    expert_probs      = None
    cascade_triggered = False

    if cascade and pred_class in CASCADE_TRIGGER:
        if expert_path is None:
            expert_path = os.environ.get("EXPERT_MODEL_PATH", "skinxai_expert_best.pth")

        if "expert" not in _model_cache:
            load_model(expert_path, "expert")

        model_exp, dev_exp = _model_cache["expert"]

        if use_tta:
            exp_raw = _infer_tta(model_exp, dev_exp, img)
        else:
            exp_raw = _infer_single(model_exp, dev_exp, img, tfm)

        # Expert 3-class: ['ak', 'bkl', 'scc']
        expert_probs = {EXPERT_CLASS_NAMES[i]: round(float(exp_raw[i]), 4)
                        for i in range(len(EXPERT_CLASS_NAMES))}

        exp_pred_idx   = int(np.argmax(exp_raw))
        exp_pred_class = EXPERT_CLASS_NAMES[exp_pred_idx]
        exp_confidence = float(exp_raw[exp_pred_idx])

        # Cascade kararı: hard override yerine soft blend
        if exp_confidence >= 0.40:
            alpha = 0.6  # expert modele ne kadar güveneceğimiz
            for cls in EXPERT_CLASS_NAMES:  # ak, bkl, scc
                idx = CLASS_NAMES.index(cls)
                probs[idx] = (1 - alpha) * probs[idx] + alpha * expert_probs[cls]
            probs = probs / probs.sum()

            pred_idx   = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            cascade_triggered = True
            mode = "cascade"

    # ── Güven skoru ───────────────────────────────────────────
    confidence = float(probs[pred_idx])

    # ── Riskli sınıf uyarıları (Stage 1 olasılıklarına göre) ─
    risk_warnings = []
    for risk_cls, thr in RISK_THRESHOLDS.items():
        risk_prob = float(probs[CLASS_NAMES.index(risk_cls)])
        if risk_prob >= thr and risk_cls != pred_class:
            risk_warnings.append((risk_cls, risk_prob))
    risk_warnings.sort(key=lambda x: -x[1])

    # sensitive_mode: en yüksek threshold'u geçen riskli sınıfa override
    if sensitive_mode and risk_warnings:
        pred_class = risk_warnings[0][0]
        pred_idx   = CLASS_NAMES.index(pred_class)
        risk_warnings = []

    all_probs = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(8)}
    is_risk   = pred_class in HIGH_RISK
    uncertain = confidence < 0.50

    # ── Uyarı mesajı ─────────────────────────────────────────
    if is_risk:
        warning = (
            f"⚠️ Bu lezyon riskli görünüyor ({CLASS_INFO[pred_class]}). "
            "Lütfen bir dermatoloğa danışın."
        )
    elif risk_warnings:
        top_cls, top_prob = risk_warnings[0]
        warning = (
            f"⚠️ Tespit: {CLASS_INFO[pred_class]} — "
            f"ancak {CLASS_INFO[top_cls]} olasılığı %{top_prob*100:.0f} seviyesinde. "
            "Bir dermatoloğa danışmanızı öneririz."
        )
    elif uncertain:
        warning = "🔍 Model bu görüntüden emin değil. Daha net bir fotoğraf deneyin veya uzman görüşü alın."
    else:
        warning = None

    return {
        "predicted_class":   pred_class,
        "full_name":         CLASS_INFO[pred_class],
        "confidence":        confidence,
        "confidence_pct":    f"{confidence:.1%}",
        "is_high_risk":      is_risk,
        "is_uncertain":      uncertain,
        "risk_warnings":     risk_warnings,
        "all_probs":         all_probs,
        "expert_probs":      expert_probs,
        "warning":           warning,
        "mode":              mode,
        "cascade_triggered": cascade_triggered,
    }


# ============================================================
# ÖRNEK KULLANIM / CLI
# ============================================================
if __name__ == "__main__":
    import sys

    usage = """
Kullanım:
  Tek model      : python predict.py <model.pth> <görüntü.jpg> [--tta] [--sensitive]
  Ensemble       : python predict.py --ensemble <v3.pth> <v6.pth> <görüntü.jpg> [--tta]
  Cascade (tam)  : python predict.py --cascade <v3.pth> <v6.pth> <expert.pth> <görüntü.jpg> [--tta]

  Env değişkenleri ile:
    V3_MODEL_PATH=... V6_MODEL_PATH=... EXPERT_MODEL_PATH=...
    python predict.py --cascade <görüntü.jpg> [--tta]

Örnek:
  python predict.py skinxai_v3_best.pth lezyon.jpg
  python predict.py --ensemble skinxai_v3_best.pth skinxai_v6_best.pth lezyon.jpg --tta
  python predict.py --cascade skinxai_v3_best.pth skinxai_v6_best.pth skinxai_expert_best.pth lezyon.jpg --tta
"""

    args = sys.argv[1:]
    if not args:
        print(usage); sys.exit(1)

    use_tta     = "--tta"       in args
    sensitive   = "--sensitive" in args
    is_ensemble = "--ensemble"  in args
    is_cascade  = "--cascade"   in args
    clean_args  = [a for a in args if not a.startswith("--")]

    result = None

    if is_cascade:
        if len(clean_args) == 4:
            v3_p, v6_p, exp_p, img_p = clean_args
        elif len(clean_args) == 1:
            v3_p  = os.environ.get("V3_MODEL_PATH",     "skinxai_v3_best.pth")
            v6_p  = os.environ.get("V6_MODEL_PATH",     "skinxai_v6_best.pth")
            exp_p = os.environ.get("EXPERT_MODEL_PATH", "skinxai_expert_best.pth")
            img_p = clean_args[0]
        else:
            print(usage); sys.exit(1)
        result = predict_image(
            img_p, v3_path=v3_p, v6_path=v6_p, expert_path=exp_p,
            use_tta=use_tta, sensitive_mode=sensitive, ensemble=True, cascade=True,
        )

    elif is_ensemble:
        if len(clean_args) == 3:
            v3_p, v6_p, img_p = clean_args
        elif len(clean_args) == 1:
            v3_p = os.environ.get("V3_MODEL_PATH", "skinxai_v3_best.pth")
            v6_p = os.environ.get("V6_MODEL_PATH", "skinxai_v6_best.pth")
            img_p = clean_args[0]
        else:
            print(usage); sys.exit(1)
        result = predict_image(
            img_p, v3_path=v3_p, v6_path=v6_p,
            use_tta=use_tta, sensitive_mode=sensitive, ensemble=True,
        )

    else:
        if len(clean_args) < 2:
            print(usage); sys.exit(1)
        model_p, img_p = clean_args[0], clean_args[1]
        result = predict_image(
            img_p, model_path=model_p,
            use_tta=use_tta, sensitive_mode=sensitive, ensemble=False,
        )

    # ── Çıktı ─────────────────────────────────────────────────
    print("\n" + "="*58)
    print("TAHMİN SONUCU  [" + result["mode"].upper() + "]")
    print("="*58)
    print(f"Sınıf     : {result['predicted_class']}")
    print(f"Tam Ad    : {result['full_name']}")
    print(f"Güven     : {result['confidence_pct']}")
    print(f"Riskli mi : {'Evet ⚠️' if result['is_high_risk'] else 'Hayır ✅'}")
    if result["cascade_triggered"]:
        print(f"Cascade   : Tetiklendi (ak/bkl/scc rafine edildi)")
    if result["warning"]:
        print(f"\n{result['warning']}")

    print("\nTüm Olasılıklar (Stage 1):")
    for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar  = "█" * int(prob * 32)
        flag = "⚠️" if cls in HIGH_RISK and prob >= RISK_THRESHOLDS.get(cls, 1.0) else ""
        print(f"  {cls:10s}: {prob:.3f}  {bar} {flag}")

    if result["expert_probs"]:
        print("\nExpert Model (ak/bkl/scc):")
        for cls, prob in sorted(result["expert_probs"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 32)
            print(f"  {cls:10s}: {prob:.3f}  {bar}")
