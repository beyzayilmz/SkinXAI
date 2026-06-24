"""
SkinXAI — Açıklanabilirlik (XAI) Modülü  ·  Grad-CAM
=====================================================
Modelin bir tahmini verirken görüntünün HANGİ bölgelerine baktığını
ısı haritası (heatmap) olarak görselleştirir. Klinik karar destek
sistemlerinde "neden bu tahmin?" sorusuna görsel cevap verir.

Kullanım
--------
    from predict import load_model
    from gradcam import generate_gradcam

    model, _ = load_model("skinxai_v6_final.pth", "v6")
    out = generate_gradcam(model, "lezyon.jpg")          # tahmin edilen sınıfa göre
    out = generate_gradcam(model, "lezyon.jpg", class_name="melanoma")

    out["overlay"]    # H×W×3 RGB  — orijinal görüntü + ısı haritası
    out["heatmap"]    # H×W×3 RGB  — sadece ısı haritası
    out["cam"]        # H×W  float [0,1] — ham aktivasyon haritası
    out["class_name"] # "melanoma" — hangi sınıf için açıklandığı

Yöntem
------
Grad-CAM (Gradient-weighted Class Activation Mapping):
  1. Hedef sınıfın skorunu son evrişim katmanına kadar geri yay (backprop).
  2. Her kanalın gradyan ortalamasını ağırlık olarak al.
  3. Aktivasyon haritalarını bu ağırlıklarla topla → ReLU → normalize.
EfficientNetV2-S'te hedef katman olarak son evrişim çıkışı (act2/conv_head,
7×7 uzamsal harita) kullanılır.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from predict import _base_tfm, _img_to_numpy, CLASS_NAMES


# HEDEF KATMAN SEÇİMİ

def _find_target_layer(model):
    """
    Grad-CAM için son uzamsal (H×W) evrişim katmanını bulur.
    EfficientNetV2-S: act2 (son aktivasyon) → bn2 → conv_head sırasıyla denenir.
    """
    backbone = getattr(model, "backbone", model)
    for name in ("act2", "bn2", "conv_head"):
        layer = getattr(backbone, name, None)
        if layer is not None:
            return layer
    # Son çare: en son blok
    blocks = getattr(backbone, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        return blocks[-1]
    raise RuntimeError("Grad-CAM için uygun hedef katman bulunamadı.")


# GRAD-CAM MOTORU

class GradCAM:
    """
    Bir modele forward/backward hook bağlayıp Grad-CAM haritası üretir.
    Kullandıktan sonra remove() çağrılmalı (with bloğu önerilir).
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or _find_target_layer(model)
        self.activations = None
        self.gradients = None
        self._fh = self.target_layer.register_forward_hook(self._forward_hook)
        self._bh = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, input_tensor, class_idx=None):
        """
        input_tensor : (1, 3, H, W) — normalize edilmiş
        class_idx    : None → argmax (modelin kendi tahmini)
        Döner: (cam[H',W'] float[0,1], kullanılan_class_idx)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        # gradyan ağırlıkları: kanal başına global ortalama
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,H',W')
        cam = F.relu(cam)[0, 0].cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx


def generate_gradcam(model, image_input, class_name=None, class_idx=None,
                     alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Tek çağrıda Grad-CAM overlay üretir.

    Parametreler
    ------------
    model       : yüklü v6/v3 modeli (predict.load_model çıktısı)
    image_input : dosya yolu | PIL.Image | numpy RGB
    class_name  : "melanoma" gibi — None ise modelin tahmin ettiği sınıf
    class_idx   : doğrudan indeks (class_name'e göre önceliklidir değilse)
    alpha       : ısı haritası saydamlığı (0=sadece görüntü, 1=sadece harita)

    Döner: dict(overlay, heatmap, cam, class_idx, class_name)
    """
    if class_name is not None and class_idx is None:
        class_idx = CLASS_NAMES.index(class_name)

    img_rgb = _img_to_numpy(image_input)            # H×W×3 RGB (uint8)
    tfm = _base_tfm()
    tensor = tfm(image=img_rgb)["image"].unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with GradCAM(model) as cam_engine:
        cam, used_idx = cam_engine(tensor, class_idx)

    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * img_rgb.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return {
        "overlay": overlay,
        "heatmap": heatmap,
        "cam": cam_resized,
        "class_idx": used_idx,
        "class_name": CLASS_NAMES[used_idx],
    }



# CLI — hızlı test

if __name__ == "__main__":
    import sys
    from predict import load_model

    if len(sys.argv) < 3:
        print("Kullanım: python gradcam.py <model.pth> <görüntü.jpg> [sınıf] [çıktı.png]")
        sys.exit(1)

    model_path, img_path = sys.argv[1], sys.argv[2]
    cls = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] in CLASS_NAMES else None
    out_path = sys.argv[4] if len(sys.argv) > 4 else "gradcam_out.png"

    version = "v6" if "v6" in model_path else ("v3" if "v3" in model_path else "v6")
    model, _ = load_model(model_path, version)
    res = generate_gradcam(model, img_path, class_name=cls)

    cv2.imwrite(out_path, cv2.cvtColor(res["overlay"], cv2.COLOR_RGB2BGR))
    print(f"[Grad-CAM] Açıklanan sınıf: {res['class_name']}")
    print(f"[Grad-CAM] Overlay kaydedildi → {out_path}")
