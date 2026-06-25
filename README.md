# SkinXAI — Klinik Karar Destek Sistemi ve Cilt Lezyonu Sınıflandırma Platformu

SkinXAI, dermatoskopik görüntüleri yapay zeka ile inceleyerek **8 farklı cilt lezyonu türünü** sınıflandıran, hekime hızlı ve **açıklanabilir** bir ön değerlendirme sunan bir **klinik karar destek aracıdır**. Amacı kesin teşhis koymak değil; riskli lezyonların gözden kaçmasını azaltarak hekime ikinci bir görüş sağlamaktır.

  **Canlı Demo:** https://skinxai.streamlit.app/
  **Repo:** https://github.com/beyzayilmz/SkinXAI

>   Bu yazılım bir karar destek aracıdır; kesin tanı yerine geçmez. Nihai tanı ve tedavi kararı hekime aittir.

---

## Öne Çıkan Özellikler

- **8 sınıflı sınıflandırma:** AK, BCC, BKL, DF, Melanoma, Nevüs, SCC, VASC
- **Açıklanabilirlik (Grad-CAM):** Modelin kararı verirken görüntünün hangi bölgesine baktığını ısı haritasıyla gösterir.
- **Girdi denetimi (OOD):** Cilt lezyonu olmayan görüntüleri (yüz, nesne, ekran görüntüsü, belge vb.) tespit edip uyarır; her görüntüyü zorla sınıflandırmaz.
- **Risk uyarısı:** Argmax tahmini farklı olsa bile, riskli bir lezyon olasılığı düşük bir eşiği aşarsa hekimi uyarır (yalancı negatifi en aza indirme önceliği).
- **TTA (Test-Time Augmentation):** 5 dönüşüm ortalamasıyla daha kararlı tahmin (isteğe bağlı).
- **İnteraktif arayüz:** Streamlit tabanlı, koyu temalı, olasılık dağılımı grafiği ile.

---

## Sınıflandırılan Lezyonlar

| Kod | Tür | Risk |
|-----|-----|------|
| melanoma | Melanoma | Yüksek (malign) |
| bcc | Bazal Hücreli Karsinom | Yüksek (malign) |
| scc | Skuamöz Hücreli Karsinom | Yüksek (malign) |
| ak | Aktinik Keratoz | Prekanseröz |
| nevus | Melanositik Nevüs (Ben) | İyi huylu |
| bkl | Benign Keratoz | İyi huylu |
| df | Dermatofibrom | İyi huylu |
| vasc | Vasküler Lezyon | İyi huylu |

---

## Kurulum (Yerel / Offline)

Gereksinim: Python 3.9–3.13.

```bash
# 1. Repoyu klonla
git clone https://github.com/beyzayilmz/SkinXAI.git
cd SkinXAI

# 2. Sanal ortam oluştur ve etkinleştir
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Uygulamayı başlat
streamlit run arayuz.py
```

Tarayıcıda otomatik olarak `http://localhost:8501` açılır.

> Model ağırlıkları (`skinxai_v6_final.pth`) ve eğitilmiş OOD kapısı (`skin_gate.npz`) repo ile birlikte gelir; ek indirme veya internet bağlantısı gerekmez (`pretrained=False`).

---

## Kullanım

1. **Analiz Paneli** sekmesinde örnek görsellerden birini seç veya kendi dermatoskopik görüntünü yükle.
2. İstersen sol menüden **TTA**'yı aç (daha doğru, ~5× yavaş).
3. **Görseli Analiz Et**'e bas.
4. Sonuç ekranında: tahmin + güven skoru, risk uyarısı, olasılık dağılımı grafiği ve **Grad-CAM odak haritası** görüntülenir.

Cilt lezyonu olmayan bir görüntü yüklenirse sistem sınıflandırma yapmaz, uygun olmadığına dair uyarı verir.

### Komut satırından tahmin

```bash
python predict.py skinxai_v6_final.pth lezyon.jpg
```

---

## Model ve Performans

- **Mimari:** EfficientNetV2-S (timm, ImageNet-21K ön-eğitimli, ~20.8M parametre) + özel sınıflandırma başlığı (v6).
- **Eğitim verisi:** HAM10000 + ISIC 2019 + ISIC 2020 (~41.000 dermoskopi görüntüsü).
- **Eğitim:** İki fazlı progressive fine-tuning; sınıf dengesizliği için ağırlıklandırılmış kayıp + hedefli veri artırma.
- **Bağımsız test sonucu (n=2293):** Accuracy **%82.0**, Macro F1 **0.702**.
- **Bilinen sınır:** Melanoma ↔ Nevüs ve AK/BKL/SCC ayrımları görsel benzerlik nedeniyle en zorlu noktalardır; düşük olasılık eşikli risk uyarı katmanı bu riski kısmen telafi eder.

### OOD Girdi Kapısı

Softmax sınıflandırıcı her görüntüyü zorla 8 sınıftan birine atar (kapalı-küme sorunu). Bunu önlemek için sınıflandırma öncesi iki katmanlı bir girdi denetimi eklenmiştir:

1. **Renk/parlaklık sezgisi** — karanlık, gri tonlu veya cilt rengi içermeyen görselleri eler.
2. **Eğitilmiş ikili kapı** — dondurulmuş v6 omurgası üzerine eğitilmiş "cilt-lezyonu / değil" lojistik regresyonu (linear probe). Lezyon-güvenli kalibre edilmiştir (gerçek lezyonu reddetme riski en aza indirilir).

Bağımsız testte (400 lezyon + 400 cilt-dışı görsel) gerçek lezyonların tamamı korunurken cilt-dışı görseller reddedilmiştir.

---

## Proje Yapısı

```
SkinXAI/
├── arayuz.py              # Streamlit arayüzü
├── predict.py             # Çıkarım + risk eşikleri + OOD girdi denetimi
├── gradcam.py             # Grad-CAM açıklanabilirlik
├── train_skin_gate.py     # OOD kapısı eğitim scripti
├── eval_gate.py           # OOD kapısı değerlendirme + grafikler
├── test_gate.py           # OOD kapısı hızlı test
├── skin_gate.npz          # Eğitilmiş OOD kapısı ağırlıkları
├── skinxai_v6_final.pth   # Üretim modeli (v6)
├── samples/               # Örnek lezyon görselleri
├── assets/                # Grafikler (confusion matrix, eğitim eğrileri vb.)
└── requirements.txt
```

---

## Teknoloji

Python · PyTorch · timm · Albumentations · OpenCV · Streamlit · Plotly

---

## Geliştirici Ekip — Datadoks

- **Beyza Yılmaz** — Model Eğitimi, Veri İşleme, Backend Entegrasyonu
- **Fatma Gizem İnanbak** — Model Eğitimi, Veri İşleme
- **Dilara Kuloğlu** — Yapay Zeka Model Mimarisi, Grad-CAM Açıklanabilirlik, OOD Girdi Denetimi
- **Hatice Ayten Kızılkaya** — Arayüz Tasarımı, Kullanıcı Deneyimi, Süreç Optimizasyonu

TeknoLAB 2025–2026 Dönem Sonu Projesi

---

## Kaynaklar

- HAM10000: https://doi.org/10.7910/DVN/DBW86T
- ISIC Archive: https://www.isic-archive.com/

## Lisans

Bu proje eğitim amaçlıdır. Tıbbi kullanım için onaylanmış bir cihaz değildir.
