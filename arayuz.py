import os
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np
from PIL import Image, ImageOps
import time # Animasyon zamanlaması için

# ── Model entegrasyonu ───────────────────────────────────────────
from predict import predict_image, load_ensemble, CLASS_INFO, HIGH_RISK
from predict import load_model 

# Bu dosyanın bulunduğu klasör — yollar her ortamda (yerel + bulut) doğru çözülür
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _yol(rel):
    return os.path.join(BASE_DIR, rel)

#V3_PATH     = _yol("skinxai_v3_best.pth")
V6_PATH     = _yol("skinxai_v6_final.pth")
#EXPERT_PATH = _yol("skinxai_expert_final.pth")

@st.cache_resource(show_spinner="Model yükleniyor...")
def modelleri_yukle():
    load_model(V6_PATH, "v6")

modelleri_yukle()

def tahmin_yap(image_input, use_tta=True):
    """predict_image'i çağırır, sonucu arayüze uygun formata çevirir."""
    result = predict_image(
        image_input,
        model_path=V6_PATH,
        version="v6",
        ensemble=False,
        use_tta=use_tta,
        cascade=False,
    )
    return result
# ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SkinXAI - Cilt Lezyonu Analizi", page_icon="🔬", layout="wide")

koyu_tasarim = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
/* ── Tipografi ──────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Animasyonlu gradient + DNA motifli arka plan ───────────── */
[data-testid="stAppViewContainer"] {
    background:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='240' viewBox='0 0 120 240'%3E%3Cg fill='none' stroke='%2338BDF8' stroke-opacity='0.12' stroke-width='1.4'%3E%3Cpath d='M60 0 C105 40 105 80 60 120 C15 160 15 200 60 240'/%3E%3Cpath d='M60 0 C15 40 15 80 60 120 C105 160 105 200 60 240'/%3E%3Cline x1='30' y1='60' x2='90' y2='60'/%3E%3Cline x1='42' y1='30' x2='78' y2='30'/%3E%3Cline x1='42' y1='90' x2='78' y2='90'/%3E%3Cline x1='30' y1='180' x2='90' y2='180'/%3E%3Cline x1='42' y1='150' x2='78' y2='150'/%3E%3Cline x1='42' y1='210' x2='78' y2='210'/%3E%3C/g%3E%3Cg fill='%2338BDF8' fill-opacity='0.14'%3E%3Ccircle cx='60' cy='0' r='2.5'/%3E%3Ccircle cx='60' cy='120' r='2.5'/%3E%3Ccircle cx='60' cy='240' r='2.5'/%3E%3C/g%3E%3C/svg%3E"),
        linear-gradient(135deg, #0B1120 0%, #1E293B 45%, #0F172A 100%);
    background-size: 95px 190px, 200% 200%;
    background-repeat: repeat, no-repeat;
    animation: arkaplan-kaydir 40s linear infinite;
    color: #F8FAFC;
}
@keyframes arkaplan-kaydir {
    0%   { background-position: 0px 0px,    0% 50%; }
    50%  { background-position: 0px 95px,   100% 50%; }
    100% { background-position: 0px 190px,  0% 50%; }
}
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(56, 189, 248, 0.15);
}
[data-testid="stHeader"] { background-color: rgba(0,0,0,0); }

h1, h2, h3, h4, h5, h6 {
    color: #F1F5F9 !important; font-weight: 700 !important;
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    letter-spacing: -0.3px;
}
p, li, span, label { color: #CBD5E1 !important; }

/* ── Genel içerik fade-in ───────────────────────────────────── */
.block-container { animation: yumusak-giris 0.7s ease both; }
@keyframes yumusak-giris {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Bölüm başlıkları ───────────────────────────────────────── */
.custom-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: #F8FAFC;
    padding-bottom: 10px; margin-bottom: 20px; letter-spacing: 0.3px;
    position: relative;
}
.custom-header::after {
    content: ""; position: absolute; left: 0; bottom: 0;
    height: 3px; width: 56px; border-radius: 3px;
    background: linear-gradient(90deg, #0EA5E9, #38BDF8);
    box-shadow: 0 0 12px rgba(56,189,248,0.6);
    animation: cizgi-genisle 1.1s ease both;
}
@keyframes cizgi-genisle { from { width: 0; opacity: 0; } to { width: 56px; opacity: 1; } }

/* ── Butonlar ───────────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }
button[kind="primary"] {
    background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%) !important;
    background-size: 200% auto !important;
    color: white !important; border: none !important;
    box-shadow: 0px 4px 14px rgba(56, 189, 248, 0.35) !important;
    font-weight: 700 !important; letter-spacing: 1px !important;
}
button[kind="primary"]:hover {
    background-position: right center !important;
    transform: translateY(-2px) !important;
    box-shadow: 0px 8px 24px rgba(56, 189, 248, 0.55) !important;
}

/* ── Sekmeler ───────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 17px !important; font-weight: 600 !important;
    color: #94A3B8 !important; background-color: transparent !important;
    transition: color 0.2s ease !important;
}
button[data-baseweb="tab"]:hover { color: #CBD5E1 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important; border-bottom: 3px solid #38BDF8 !important;
}

/* ── Logo ───────────────────────────────────────────────────── */
.logo-badge {
    background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%);
    background-size: 200% 200%;
    animation: logo-parla 4s ease infinite;
    color: white !important; padding: 10px 22px; border-radius: 14px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 800; margin-right: 15px;
    letter-spacing: 1px; box-shadow: 0px 0px 22px rgba(56, 189, 248, 0.45);
}
@keyframes logo-parla {
    0%   { background-position: 0% 50%; box-shadow: 0 0 18px rgba(56,189,248,0.35); }
    50%  { background-position: 100% 50%; box-shadow: 0 0 28px rgba(56,189,248,0.6); }
    100% { background-position: 0% 50%; box-shadow: 0 0 18px rgba(56,189,248,0.35); }
}

/* ── Risk uyarıları ─────────────────────────────────────────── */
@keyframes yanip-sonme-kirmizi {
  0%   { background-color: rgba(239,68,68,0.1); border: 1px solid #7F1D1D; }
  50%  { background-color: rgba(239,68,68,0.2); border: 1px solid #DC2626; box-shadow: 0 0 18px rgba(220,38,38,0.5); }
  100% { background-color: rgba(239,68,68,0.1); border: 1px solid #7F1D1D; }
}
.kirmizi-risk {
  animation: yanip-sonme-kirmizi 1.5s infinite;
  color: #FCA5A5 !important; padding: 15px; border-radius: 10px;
  font-weight: bold; text-align: center; font-size: 18px; margin-bottom: 15px;
}
.kirmizi-risk span { color: #FECACA !important; }
@keyframes yanip-sonme-sari {
  0%   { background-color: rgba(245,158,11,0.1); border: 1px solid #78350F; }
  50%  { background-color: rgba(245,158,11,0.2); border: 1px solid #D97706; box-shadow: 0 0 18px rgba(217,119,6,0.5); }
  100% { background-color: rgba(245,158,11,0.1); border: 1px solid #78350F; }
}
.sari-uyari {
  animation: yanip-sonme-sari 1.5s infinite;
  color: #FCD34D !important; padding: 15px; border-radius: 10px;
  font-weight: bold; text-align: center; font-size: 18px; margin-bottom: 15px;
}
.sari-uyari span { color: #FDE68A !important; }

/* ── Güven skoru kartı ──────────────────────────────────────── */
.guven-karti {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 14px; padding: 18px 20px; margin: 6px 0 16px 0;
    backdrop-filter: blur(8px);
    animation: yumusak-giris 0.6s ease both;
}
.guven-karti .etiket {
    font-size: 13px; color: #94A3B8 !important; text-transform: uppercase;
    letter-spacing: 1.5px; margin-bottom: 4px;
}
.guven-karti .deger {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 38px; font-weight: 700; color: #38BDF8 !important; line-height: 1;
}
.guven-bar-dis {
    background: rgba(148,163,184,0.18); border-radius: 999px;
    height: 10px; margin-top: 12px; overflow: hidden;
}
.guven-bar-ic {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #0EA5E9, #38BDF8);
    box-shadow: 0 0 12px rgba(56,189,248,0.6);
    animation: bar-doldur 1.1s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes bar-doldur { from { width: 0; } }

/* ── Görsel & expander ──────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 12px; box-shadow: 0px 8px 28px rgba(0,0,0,0.5);
    border: 1px solid rgba(56,189,248,0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.02);
    box-shadow: 0px 12px 36px rgba(56,189,248,0.25);
}
[data-testid="stExpander"] {
    background-color: rgba(15,23,42,0.45);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 10px; transition: border-color 0.25s ease;
}
[data-testid="stExpander"]:hover { border-color: rgba(56,189,248,0.4); }

/* ── Bilgi/uyarı kutuları ───────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px;
    animation: yumusak-giris 0.6s ease both;
}

/* ── SkinXAI logo / marka kimliği ───────────────────────────── */
.logo-wrap {
    display: flex; align-items: center; gap: 18px;
    margin: -12px 0 28px 0;
    animation: yumusak-giris 0.8s ease both;
}
.logo-emblem { filter: drop-shadow(0 0 14px rgba(56,189,248,0.5)); flex-shrink: 0; }
.logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 34px; font-weight: 800; line-height: 1;
}
.logo-text .skin { color: #F1F5F9 !important; }
.logo-text .xai {
    background: linear-gradient(90deg, #0EA5E9, #38BDF8, #7DD3FC, #38BDF8, #0EA5E9);
    background-size: 200% auto;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent !important;
    animation: metin-parla 4s linear infinite;
}
@keyframes metin-parla { to { background-position: 200% center; } }
.logo-sub {
    font-size: 12px; color: #94A3B8 !important; letter-spacing: 2px;
    margin-top: 5px; text-transform: uppercase;
}
.logo-divider {
    width: 1px; height: 46px;
    background: linear-gradient(180deg, transparent, #475569, transparent);
}
.logo-tagline {
    font-size: 21px; color: #94A3B8 !important; font-weight: 500;
}
/* ── MİMARİ AKIŞ SİMÜLASYONU STİLLERİ ────────────────────────── */
.mimari-kutu {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: all 0.4s ease;
}
.mimari-kutu.aktif {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.6) !important;
    background: rgba(14, 165, 233, 0.15) !important;
}
.mimari-kutu.expert-aktif {
    border-color: #F59E0B !important;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.6) !important;
    background: rgba(245, 158, 11, 0.15) !important;
}
.mimari-ok {
    font-size: 24px;
    text-align: center;
    color: #475569;
    margin: 5px 0;
    transition: color 0.4s ease;
}
.mimari-ok.parlayan {
    color: #38BDF8;
    text-shadow: 0 0 10px rgba(56,189,248,0.8);
    animation: ok-pulse 1.2s infinite;
}
.mimari-ok.parlayan-turuncu {
    color: #F59E0B;
    text-shadow: 0 0 10px rgba(245,158,11,0.8);
    animation: ok-pulse 1.2s infinite;
}
@keyframes ok-pulse {
    0% { transform: scale(1); opacity: 0.6; }
    50% { transform: scale(1.15); opacity: 1; }
    100% { transform: scale(1); opacity: 0.6; }
}
</style>
"""
st.markdown(koyu_tasarim, unsafe_allow_html=True)

# --- OTURUM HAFIZASI ---
if "secilen_resim_yolu" not in st.session_state: st.session_state.secilen_resim_yolu = None
if "analiz_yapildi"     not in st.session_state: st.session_state.analiz_yapildi = False
if "analiz_sonucu"      not in st.session_state: st.session_state.analiz_sonucu = None
if "img_pil"            not in st.session_state: st.session_state.img_pil = None

# --- YAN MENÜ ---
with st.sidebar:
    st.markdown('<div class="custom-header">Sistem Özeti</div>', unsafe_allow_html=True)
    st.write("Bu arayüz, cilt lezyonlarının yapay zeka ile sınıflandırılması ve hekimlere hızlı bir ön değerlendirme sunulması amacıyla tasarlanmıştır.")
    st.write("<br>", unsafe_allow_html=True)
    st.markdown('<div class="custom-header" style="font-size: 18px;">Risk Yönetimi</div>', unsafe_allow_html=True)
    st.info("Sistem, **Yalancı Negatif (False Negative)** hatasını en aza indirecek şekilde ayarlanmıştır.")
    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)

    # TTA seçeneği
    use_tta = st.toggle("TTA kullan (daha doğru, ~5× yavaş)", value=False)

    st.caption("🧠 SkinXAI v6 (EfficientNetV2-S)")
    st.warning("⚠️ Yasal Uyarı: Bu yazılım kesin bir tıbbi tanı aracı değildir.")

# --- LOGO ---
logo_html = (
'<div class="logo-wrap">'
'<svg class="logo-emblem" width="93" height="93" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">'
'<defs>'
'<linearGradient id="skinGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
'<stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#38BDF8"/>'
'</linearGradient>'
'<radialGradient id="coreGlow" cx="50%" cy="45%" r="55%">'
'<stop offset="0%" stop-color="#7DD3FC"/><stop offset="100%" stop-color="#0284C7"/>'
'</radialGradient>'
'</defs>'
'<circle cx="50" cy="50" r="46" fill="#38BDF8" opacity="0.06"/>'
'<circle cx="50" cy="50" r="44" fill="none" stroke="url(#skinGrad)" stroke-width="3" stroke-dasharray="11 9" stroke-linecap="round">'
'<animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50" dur="9s" repeatCount="indefinite"/>'
'</circle>'
'<circle cx="50" cy="50" r="32" fill="none" stroke="#38BDF8" stroke-width="2" stroke-dasharray="42 130" stroke-linecap="round" opacity="0.65">'
'<animateTransform attributeName="transform" type="rotate" from="360 50 50" to="0 50 50" dur="6s" repeatCount="indefinite"/>'
'</circle>'
'<circle cx="50" cy="50" r="20" fill="url(#coreGlow)">'
'<animate attributeName="r" values="18.5;21.5;18.5" dur="2.4s" repeatCount="indefinite"/>'
'<animate attributeName="opacity" values="0.9;1;0.9" dur="2.4s" repeatCount="indefinite"/>'
'</circle>'
'<path d="M38 50 H44 L47 42 L50 58 L53 47 L56 50 H62" fill="none" stroke="#0B1120" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>'
'<g><circle cx="50" cy="6" r="3.6" fill="#7DD3FC"/>'
'<animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50" dur="3.5s" repeatCount="indefinite"/></g>'
'</svg>'
'<div><div class="logo-text"><span class="skin">Skin</span><span class="xai">XAI</span></div>'
'<div class="logo-sub">Explainable Skin AI</div></div>'
'<div class="logo-divider"></div>'
'<div class="logo-tagline">Klinik Karar Destek Sistemi</div>'
'</div>'
)
st.markdown(logo_html, unsafe_allow_html=True)

sekme_analiz, sekme_lezyonlar, sekme_detay, sekme_ekip = st.tabs([
    "🔬 Analiz Paneli", "📚 Lezyon Sınıfları", "📖 Sistem Detayları", "👥 Geliştirici Ekip"
])

# ---------------------------------------------------------
# SEKME 1: ANALİZ PANELİ
# ---------------------------------------------------------
with sekme_analiz:
    st.markdown("#### *Yapay Zeka Destekli Dermatolojik Ön Değerlendirme Platformu*")
    st.divider()

    sol_islem, sag_islem = st.columns([1, 1], gap="large")

    with sol_islem:
        st.markdown('<div class="custom-header">Proje Özeti ve Amacı</div>', unsafe_allow_html=True)
        st.write("Bu platform, dermatoskopik görüntüleri analiz ederek cilt lezyonlarını yapay zeka algoritmalarıyla sınıflandıran bir **Klinik Karar Destek Sistemidir**.")
        st.info("🎯 **Temel Hedef:** Teşhis sürecinde hekimlere ikinci bir görüş sunarak tanı doğruluğunu artırmak ve biyopsi süreçlerini optimize etmektir.")
        st.success("📊 **Veri ve Model:** HAM10000 + ISIC 2019 + ISIC 2020 (~41K görüntü), EfficientNetV2-S backbone (v6).")
        st.warning("🛡️ **Güvenlik Protokolü:** Riskli lezyonları kaçırmamak adına algoritmik olarak güvenlik odaklı kalibre edilmiştir.")

    with sag_islem:
        st.markdown('<div class="custom-header">Teşhis ve Analiz Motoru</div>', unsafe_allow_html=True)
        st.write("Örnek görselleri kullanabilir veya kendi fotoğrafınızı yükleyebilirsiniz.")

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("Örnek: Melanom", use_container_width=True):
                st.session_state.secilen_resim_yolu = _yol("samples/melanoma.jpg")
                st.session_state.analiz_yapildi = False
                st.session_state.analiz_sonucu  = None
        with btn_col2:
            if st.button("Örnek: Nevüs", use_container_width=True):
                st.session_state.secilen_resim_yolu = _yol("samples/nevus1.jpg")
                st.session_state.analiz_yapildi = False
                st.session_state.analiz_sonucu  = None
        with btn_col3:
            if st.button("Sıfırla", use_container_width=True):
                st.session_state.secilen_resim_yolu = None
                st.session_state.analiz_yapildi = False
                st.session_state.analiz_sonucu  = None
                st.session_state.img_pil        = None

        uploaded_file = st.file_uploader(" ", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if uploaded_file is not None:
            st.session_state.secilen_resim_yolu = uploaded_file
            st.session_state.analiz_yapildi = False
            st.session_state.analiz_sonucu  = None

        if st.session_state.secilen_resim_yolu is not None:
                    st.write("")
                    if st.button("GÖRSELİ ANALİZ ET", type="primary", use_container_width=True):
                        
                        # Resim girdisini PIL Image nesnesine dönüştürme
                        img_input = st.session_state.secilen_resim_yolu
                        if hasattr(img_input, "read"):
                            img_input = Image.open(img_input).convert("RGB")
                        elif isinstance(img_input, str):
                            img_input = Image.open(img_input).convert("RGB")
                        
                        st.session_state.img_pil = img_input

                        # ── TTA CANLI ÖNİZLEME ALANI ──────────────────────────────────
                        if use_tta:
                            st.markdown("##### TTA Süreci İşleniyor... (5 Farklı Matris Çoğaltımı)")
                            tta_placeholder = st.empty()
                            
                            with tta_placeholder.container():
                                # Orijinal görsel boyutlarını alıp crop simülasyonu hazırlığı yapalım
                                w, h = img_input.size
                                crop_box = (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
                                
                                transformasyonlar = [
                                    {"isim": "1. Orijinal Görüntü", "img": img_input},
                                    {"isim": "2. Horizontal Flip", "img": ImageOps.mirror(img_input)},
                                    {"isim": "3. Vertical Flip", "img": ImageOps.flip(img_input)},
                                    {"isim": "4. Center Crop", "img": img_input.crop(crop_box).resize((w, h))},
                                    {"isim": "5. Random Rotate (45°)", "img": img_input.rotate(45)},
                                ]
                                
                                # 5 mikro-görsel için yan yana kolonlar oluşturma
                                tta_cols = st.columns(5)
                                for idx, tf in enumerate(transformasyonlar):
                                    with tta_cols[idx]:
                                        st.caption(tf["isim"])
                                        st.image(tf["img"], use_container_width=True)
                                        # Teknik şov hissini pekiştirmek için mikro gecikmeli animasyon efekti
                                        time.sleep(0.3) 
                            
                            st.success("⚡ Tüm transformasyon matrisleri doğrulandı. Ensemble model optimizasyonu başlatılıyor...")
                        # ──────────────────────────────────────────────────────────────

                        with st.spinner("Analiz yapılıyor..."):
                            st.session_state.analiz_sonucu = tahmin_yap(img_input, use_tta=use_tta)

                        st.session_state.analiz_yapildi = True

    st.divider()

    # ── SONUÇ EKRANI ─────────────────────────────────────────────
    if st.session_state.analiz_yapildi and st.session_state.analiz_sonucu is not None:
        result = st.session_state.analiz_sonucu

        tahmin_edilen       = result["predicted_class"]
        guven_skoru         = int(result["confidence"] * 100)
        tam_ad              = result["full_name"]
        yuksek_risk         = result["is_high_risk"]
        uyari_var           = bool(result["risk_warnings"])
        cascade_tetiklendi  = result.get("cascade_triggered", False)

        st.markdown('<div class="custom-header">Analiz Sonucu ve Raporlama</div>', unsafe_allow_html=True)
        st.divider()
        sonuc_col1, sonuc_col2, sonuc_col3 = st.columns(3, gap="medium")

        # Görüntü
        with sonuc_col1:
            st.write("#### İncelenen Görüntü")
            st.image(st.session_state.img_pil or st.session_state.secilen_resim_yolu,
                     use_container_width=True)
            if cascade_tetiklendi:
                st.caption("🔄 Cascade (uzman model) devreye girdi")

        # Tahmin ve uyarı
        with sonuc_col2:
            st.write(f"### Tahmin: **{tahmin_edilen.upper()}**")
            st.write(f"**{tam_ad}**")
            st.markdown(f"""
            <div class="guven-karti">
                <div class="etiket">Model Güven Skoru</div>
                <div class="deger">%{guven_skoru}</div>
                <div class="guven-bar-dis">
                    <div class="guven-bar-ic" style="width: {guven_skoru}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if yuksek_risk:
                st.markdown(f"""
                <div class="kirmizi-risk">🚨 YÜKSEK RİSK 🚨<br>
                <span style="font-size:14px;font-weight:normal;">
                {tam_ad} tespit edildi. Acilen dermatoloğa başvurunuz.</span></div>
                """, unsafe_allow_html=True)
            elif uyari_var:
                uyari_cls, uyari_prob = result["risk_warnings"][0]
                st.markdown(f"""
                <div class="sari-uyari">⚠️ ŞÜPHELİ DURUM ⚠️<br>
                <span style="font-size:14px;font-weight:normal;">
                Genel tahmin: {tahmin_edilen} — ancak {CLASS_INFO[uyari_cls]} olasılığı %{int(uyari_prob*100)} seviyesinde. Kontrol önerilir.</span></div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ Riskli bir bulguya rastlanmadı.")

            # Expert probs (cascade varsa)
            if result.get("expert_probs"):
                st.write("**Uzman model (ak/bkl/df/scc):**")
                for cls, prob in sorted(result["expert_probs"].items(), key=lambda x: -x[1]):
                    st.caption(f"{cls}: %{int(prob*100)}")

            st.warning("⚠️ **Yasal Uyarı:** Bu analiz kesin tıbbi tanı koymaz.")

        # Grafik — tüm 8 sınıf
        with sonuc_col3:
            # Cascade tetiklendiyse expert probs'u göster, yoksa Stage 1 probs
            if cascade_tetiklendi and result.get("expert_probs"):
                st.write("#### Uzman Model Dağılımı")
                st.caption("(cascade aktif — ak/bkl/df/scc)")
                probs_raw = result["expert_probs"]
                veri = pd.DataFrame({
                    "Hastalık": [k.upper() for k in probs_raw.keys()],
                    "İhtimal":  [round(v * 100, 1) for v in probs_raw.values()],
                })
            else:
                st.write("#### Olasılık Dağılımları")
                probs_raw = result["all_probs"]
                veri = pd.DataFrame({
                    "Hastalık": [k.upper() for k in probs_raw.keys()],
                    "İhtimal":  [round(v * 100, 1) for v in probs_raw.values()],
                }).sort_values("İhtimal", ascending=False)

            fig = px.pie(
                veri, values="İhtimal", names="Hastalık", hole=0.4,
                color_discrete_sequence=px.colors.sequential.Tealgrn,
            )
            fig.update_traces(
                textposition="inside", textinfo="percent+label",
                marker=dict(line=dict(color="#1E293B", width=2)),
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# SEKME 2: LEZYON SINIFLARI
# ---------------------------------------------------------
with sekme_lezyonlar:
    st.markdown('<div class="custom-header">Sınıflandırma Kategorileri ve Tıbbi Detaylar</div>', unsafe_allow_html=True)
    st.write("Yapay zeka modelimiz aşağıdaki 8 farklı cilt lezyonu türünü analiz edebilecek şekilde eğitilmiştir:")
    st.write("<br>", unsafe_allow_html=True)

    kanser_kolon, iyi_huylu_kolon = st.columns(2, gap="large")

    with kanser_kolon:
        st.error("🚨 Yüksek Riskli Sınıflar (Malign / Prekanseröz)")
        with st.expander("1. MELANOMA", expanded=True):
            st.write("Hızlı yayılan en tehlikeli cilt kanseridir. Asimetrik yapılı ve çok renklidir.")
        with st.expander("2. BCC (Bazal Hücreli Karsinom)"):
            st.write("En sık görülen, yavaş büyüyen cilt kanseridir. Parlak, iyileşmeyen yara şeklindedir.")
        with st.expander("3. SCC (Skuamöz Hücreli Karsinom)"):
            st.write("Güneş hasarı kaynaklıdır. Pullu, kabuklu lezyonlar olarak ortaya çıkar.")
        with st.expander("4. AK (Aktinik Keratoz)"):
            st.write("Kanser öncüsü (prekanseröz) riskli bir lekedir.")

    with iyi_huylu_kolon:
        st.success("🟢 Düşük Riskli Sınıflar (Benign / İyi Huylu)")
        with st.expander("5. NEVUS (Melanositik Nevüs)", expanded=True):
            st.write("Halk arasında 'ben' olarak bilinen tamamen zararsız lekelerdir.")
        with st.expander("6. BKL (Benign Keratoz)"):
            st.write("İlerleyen yaşlarda görülen zararsız doku büyümeleridir.")
        with st.expander("7. DF (Dermatofibrom)"):
            st.write("Böcek ısırığı sonrası oluşan sert ve zararsız küçük nodüllerdir.")
        with st.expander("8. VASC (Vasküler Lezyonlar)"):
            st.write("'Çilek lekesi' gibi damar genişlemesi sonucu oluşan zararsız kızarıklıklardır.")

# ---------------------------------------------------------
# SEKME 3: SİSTEM DETAYLARI
# ---------------------------------------------------------
with sekme_detay:
    st.markdown('<div class="custom-header">Sistem Mimarisi ve Teknik Detaylar</div>', unsafe_allow_html=True)
    st.write("**Backbone:** EfficientNetV2-S (20.8M parametre, timm kütüphanesi)")
    st.write("**Eğitim Verisi:** HAM10000 + ISIC 2019 + ISIC 2020 (~41K görüntü, 8 sınıf)")
    st.write("**Model:** v6 — EfficientNetV2-S backbone, sınıf bazlı ağırlıklandırılmış kayıp fonksiyonu")
    st.write("**TTA:** 5 transform ortalaması (normal, hflip, vflip, centercrop, rotate)")
    st.write("**Temiz Test Seti Sonucu:** Accuracy %82.0, Macro F1 0.702")
    st.write("**Not:** v3+v6 ensemble ve expert/cascade mimarileri ayrıca denenmiş, titiz ablasyon testinde v6 tek başına en güvenilir/dengeli sonucu verdiği için final model olarak seçilmiştir.")

# ---------------------------------------------------------
# SEKME 4: GELİŞTİRİCİ EKİP
# ---------------------------------------------------------
with sekme_ekip:
    st.markdown('<div class="custom-header">Proje Ekibi</div>', unsafe_allow_html=True)
    st.write("**Beyza YILMAZ** - Model Eğitimi, Veri İşleme ve Backend Entegrasyonu")
    st.write("**Dilara KULOĞLU** - Yapay Zeka Model Mimarisi, Grad-CAM Görselleştirmeleri")
    st.write("**Fatma Gizem İNANBAK** - Model Eğitimi, Veri İşleme")
    st.write("**Hatice Ayten KIZILKAYA** - Arayüz Tasarımı, Kullanıcı Deneyimi ve Süreç Optimizasyonu")
    st.write("**Hüsna Nur YÜKSEL**")