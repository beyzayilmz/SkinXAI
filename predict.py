"""
SkinXAI — Inference Modülü
===========================
Kullanım:
    from predict import predict_image

    result = predict_image("cilt_lezyon.jpg")
    print(result["predicted_class"])   # örn: "melanoma"
    print(result["confidence"])        # örn: 0.82
    print(result["all_probs"])         # tüm sınıf olasılıkları

Gradio / Streamlit entegrasyonu için predict_image fonksiyonunu direkt kullanabilirsin.
"""

import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from PIL import Image
import os

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

# Riskli sınıflar (kırmızı uyarı göstermek için)
HIGH_RISK = {"melanoma", "bcc", "scc", "ak"}

# ============================================================
# MODEL MİMARİSİ
# ============================================================
class SkinModel(nn.Module):
    def __init__(self, num_classes=8, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=False,      # inference'ta pretrained gerekmez
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


# ============================================================
# MODEL YÜKLEME (bir kez yükle, cache'de tut)
# ============================================================
_model = None
_device = None

def load_model(model_path: str, device: str = None):
    """
    Modeli yükler ve cache'e alır.
    İlk çağrıda yükler, sonraki çağrılarda cache'den döner.

    Args:
        model_path: .pth dosyasının yolu
        device: "cuda", "cpu" veya None (otomatik seçim)
    """
    global _model, _device

    if _model is not None:
        return _model, _device

    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)

    _model = SkinModel(num_classes=8).to(_device)
    ckpt = torch.load(model_path, map_location=_device, weights_only=False)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()

    print(f"Model yüklendi ✅  |  Cihaz: {_device}")
    return _model, _device


# ============================================================
# TRANSFORM
# ============================================================
def _get_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ============================================================
# ANA TAHMİN FONKSİYONU
# ============================================================
def predict_image(image_input, model_path: str = None, use_tta: bool = False):
    """
    Cilt lezyon görüntüsünü sınıflandırır.

    Args:
        image_input : Dosya yolu (str), numpy array (H,W,3 RGB), veya PIL Image
        model_path  : .pth dosyası yolu. None ise MODEL_PATH env değişkenine bakar.
        use_tta     : Test Time Augmentation kullan (daha yavaş ama daha doğru)

    Returns:
        dict:
            predicted_class  : "melanoma"
            full_name        : "Melanoma"
            confidence       : 0.82  (0-1 arası)
            is_high_risk     : True
            all_probs        : {"melanoma": 0.82, "nevus": 0.10, ...}
            warning          : "⚠️ Riskli lezyon..." veya None
    """
    global _model, _device

    # Model yolu
    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "skinxai_v3_best.pth")

    # Model yükle (cache'deyse tekrar yüklemez)
    model, device = load_model(model_path)

    # Görüntüyü numpy RGB array'e çevir
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img = image_input
        if img.shape[-1] == 4:   # RGBA ise
            img = img[:, :, :3]
    else:
        raise TypeError(f"Desteklenmeyen girdi tipi: {type(image_input)}")

    tfm = _get_transform()

    if use_tta:
        probs = _predict_tta(model, device, img, tfm)
    else:
        probs = _predict_single(model, device, img, tfm)

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    all_probs = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(8)}

    is_risk = pred_class in HIGH_RISK
    warning = (
        f"⚠️ Bu lezyon potansiyel olarak riskli görünüyor ({CLASS_INFO[pred_class]}). "
        "Lütfen bir dermatoloğa danışın."
        if is_risk else None
    )

    return {
        "predicted_class": pred_class,
        "full_name":       CLASS_INFO[pred_class],
        "confidence":      confidence,
        "confidence_pct":  f"{confidence:.1%}",
        "is_high_risk":    is_risk,
        "all_probs":       all_probs,
        "warning":         warning,
    }


def _predict_single(model, device, img_rgb, tfm):
    tensor = tfm(image=img_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return probs


def _predict_tta(model, device, img_rgb, tfm):
    """5 farklı augmentation ortalaması alır."""
    tta_tfms = [
        A.Compose([A.Resize(224,224), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
        A.Compose([A.Resize(224,224), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
        A.Compose([A.Resize(224,224), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
        A.Compose([A.Resize(256,256), A.CenterCrop(224,224), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
        A.Compose([A.Resize(224,224), A.Rotate(limit=15, p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
    ]
    all_probs = []
    with torch.no_grad():
        for t in tta_tfms:
            tensor = t(image=img_rgb)["image"].unsqueeze(0).to(device)
            out    = model(tensor)
            probs  = torch.softmax(out, dim=1)[0].cpu().numpy()
            all_probs.append(probs)
    return np.mean(all_probs, axis=0)


# ============================================================
# ÖRNEK KULLANIM
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Kullanım: python predict.py <model_path> <image_path>")
        print("Örnek   : python predict.py skinxai_v3_best.pth test.jpg")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    result = predict_image(image_path, model_path=model_path)

    print("\n" + "="*50)
    print("TAHMİN SONUCU")
    print("="*50)
    print(f"Sınıf      : {result['predicted_class']}")
    print(f"Tam Ad     : {result['full_name']}")
    print(f"Güven      : {result['confidence_pct']}")
    print(f"Riskli mi  : {'Evet ⚠️' if result['is_high_risk'] else 'Hayır ✅'}")
    if result["warning"]:
        print(f"\n{result['warning']}")
    print("\nTüm Olasılıklar:")
    for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:10s}: {prob:.3f}  {bar}")