import os
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np

# ── Model entegrasyonu ───────────────────────────────────────────
from predict import predict_image, load_ensemble, CLASS_INFO, HIGH_RISK

V3_PATH     = "skinxai_v3_best.pth"
V6_PATH     = "skinxai_v6_final.pth"
EXPERT_PATH = "skinxai_expert_final.pth"

@st.cache_resource(show_spinner="Modeller yükleniyor...")
def modelleri_yukle():
    expert = EXPERT_PATH if os.path.exists(EXPERT_PATH) else None
    load_ensemble(V3_PATH, V6_PATH, expert)
    return expert is not None

CASCADE_VAR = modelleri_yukle()

def tahmin_yap(image_input, use_tta=True):
    """predict_image'i çağırır, sonucu arayüze uygun formata çevirir."""
    expert = EXPERT_PATH if CASCADE_VAR else None
    result = predict_image(
        image_input,
        ensemble=True,
        v3_path=V3_PATH,
        v6_path=V6_PATH,
        expert_path=expert,
        use_tta=use_tta,
        cascade=CASCADE_VAR,
    )
    return result
# ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="DermAI - Cilt Lezyonu Analizi", layout="wide")

koyu_tasarim = """
<style>
[data-testid="stAppViewContainer"] { background-color: #1E293B; color: #F8FAFC; }
[data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #334155; }
[data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; font-weight: 600 !important; }
p, li, span, label { color: #CBD5E1 !important; }
.custom-header {
    font-size: 22px; font-weight: 600; color: #F8FAFC;
    border-bottom: 2px solid #334155; padding-bottom: 10px;
    margin-bottom: 20px; letter-spacing: 0.5px;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%) !important;
    color: white !important; border: none !important;
    box-shadow: 0px 4px 10px rgba(56, 189, 248, 0.3) !important;
    font-weight: 600 !important; letter-spacing: 1px !important;
}
button[kind="primary"]:hover {
    opacity: 0.9 !important;
    box-shadow: 0px 4px 15px rgba(56, 189, 248, 0.5) !important;
}
button[data-baseweb="tab"] {
    font-size: 18px !important; font-weight: 600 !important;
    color: #94A3B8 !important; background-color: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important; border-bottom: 3px solid #38BDF8 !important;
}
.logo-badge {
    background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%);
    color: white !important; padding: 10px 20px; border-radius: 12px;
    font-size: 28px; font-weight: 800; margin-right: 15px;
    letter-spacing: 1px; box-shadow: 0px 0px 15px rgba(56, 189, 248, 0.3);
}
@keyframes yanip-sonme-kirmizi {
  0%   { background-color: rgba(239,68,68,0.1); border: 1px solid #7F1D1D; }
  50%  { background-color: rgba(239,68,68,0.2); border: 1px solid #DC2626; box-shadow: 0 0 15px rgba(220,38,38,0.4); }
  100% { background-color: rgba(239,68,68,0.1); border: 1px solid #7F1D1D; }
}
.kirmizi-risk {
  animation: yanip-sonme-kirmizi 1.5s infinite;
  color: #FCA5A5 !important; padding: 15px; border-radius: 8px;
  font-weight: bold; text-align: center; font-size: 18px; margin-bottom: 15px;
}
.kirmizi-risk span { color: #FECACA !important; }
@keyframes yanip-sonme-sari {
  0%   { background-color: rgba(245,158,11,0.1); border: 1px solid #78350F; }
  50%  { background-color: rgba(245,158,11,0.2); border: 1px solid #D97706; box-shadow: 0 0 15px rgba(217,119,6,0.4); }
  100% { background-color: rgba(245,158,11,0.1); border: 1px solid #78350F; }
}
.sari-uyari {
  animation: yanip-sonme-sari 1.5s infinite;
  color: #FCD34D !important; padding: 15px; border-radius: 8px;
  font-weight: bold; text-align: center; font-size: 18px; margin-bottom: 15px;
}
.sari-uyari span { color: #FDE68A !important; }
[data-testid="stImage"] img { border-radius: 10px; box-shadow: 0px 4px 20px rgba(0,0,0,0.4); border: 1px solid #334155; }
[data-testid="stExpander"] { background-color: rgba(15,23,42,0.4); border: 1px solid #334155; }
</style>
"""
st.markdown(koyu_tasarim, unsafe_allow_html=True)

# --- OTURUM HAFIZASI ---
if "secilen_resim_yolu" not in st.session_state: st.session_state.secilen_resim_yolu = None
if "analiz_yapildi"     not in st.session_state: st.session_state.analiz_yapildi = False
if "analiz_sonucu"      not in st.session_state: st.session_state.analiz_sonucu = None

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

    cascade_durum = "✅ Cascade aktif" if CASCADE_VAR else "⚠️ Sadece ensemble"
    st.caption(cascade_durum)
    st.warning("⚠️ Yasal Uyarı: Bu yazılım kesin bir tıbbi tanı aracı değildir.")

# --- LOGO ---
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 25px; margin-top: -20px;">
    <div class="logo-badge">🧬 DermAI</div>
    <div style="font-size: 22px; color: #94A3B8; font-weight: 500; border-left: 2px solid #475569; padding-left: 15px;">
        Klinik Karar Destek Sistemi
    </div>
</div>
""", unsafe_allow_html=True)

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
        st.success("📊 **Veri ve Model:** HAM10000 + ISIC 2019 (41K görüntü), EfficientNetV2-S backbone, v3+v6 per-class ensemble" + (" + expert cascade" if CASCADE_VAR else "") + ".")
        st.warning("🛡️ **Güvenlik Protokolü:** Riskli lezyonları kaçırmamak adına algoritmik olarak güvenlik odaklı kalibre edilmiştir.")

    with sag_islem:
        st.markdown('<div class="custom-header">Teşhis ve Analiz Motoru</div>', unsafe_allow_html=True)
        st.write("Örnek görselleri kullanabilir veya kendi fotoğrafınızı yükleyebilirsiniz.")

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("Örnek: Melanom", use_container_width=True):
                st.session_state.secilen_resim_yolu = "melanom.jpeg"
                st.session_state.analiz_yapildi = False
                st.session_state.analiz_sonucu  = None
        with btn_col2:
            if st.button("Örnek: Nevüs", use_container_width=True):
                st.session_state.secilen_resim_yolu = "nevus.jpeg"
                st.session_state.analiz_yapildi = False
                st.session_state.analiz_sonucu  = None
        with btn_col3:
            if st.button("Sıfırla", use_container_width=True):
                st.session_state.secilen_resim_yolu = None
                st.session_state.analiz_yapildi = False
                st.session_state.analiz_sonucu  = None

        uploaded_file = st.file_uploader(" ", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if uploaded_file is not None:
            st.session_state.secilen_resim_yolu = uploaded_file
            st.session_state.analiz_yapildi = False
            st.session_state.analiz_sonucu  = None

        if st.session_state.secilen_resim_yolu is not None:
            st.write("")
            if st.button("GÖRSELİ ANALİZ ET", type="primary", use_container_width=True):
                with st.spinner("Analiz yapılıyor..."):
                    img_input = st.session_state.secilen_resim_yolu
                    # UploadedFile ise PIL Image'e çevir
                    if hasattr(img_input, "read"):
                        img_input = Image.open(img_input).convert("RGB")
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
        sonuc_col1, sonuc_col2, sonuc_col3 = st.columns(3, gap="medium")

        # Görüntü
        with sonuc_col1:
            st.write("#### İncelenen Görüntü")
            img_goster = st.session_state.secilen_resim_yolu
            if hasattr(img_goster, "read"):
                img_goster.seek(0)
            st.image(img_goster, use_container_width=True)
            if cascade_tetiklendi:
                st.caption("🔄 Cascade (uzman model) devreye girdi")

        # Tahmin ve uyarı
        with sonuc_col2:
            st.write(f"### Tahmin: **{tahmin_edilen.upper()}**")
            st.write(f"**{tam_ad}**")
            st.write(f"#### Güven: %{guven_skoru}")
            st.write("")

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
    st.write("**Eğitim Verisi:** HAM10000 + ISIC 2019 (~41K görüntü, 8 sınıf)")
    st.write("**Ensemble:** v3 + v6 per-class ağırlıklı ensemble (melanoma/ak için v3 ağır, bcc/bkl için v6 ağır)")
    st.write("**Cascade:** ak/bkl/scc/df için ayrı uzman model (SCC recall: 54% → 87.8%)")
    st.write("**TTA:** 5 transform ortalaması (normal, hflip, vflip, centercrop, rotate)")
    st.write("**Kaggle Test F1:** 0.812 (TTA, cascade)")

# ---------------------------------------------------------
# SEKME 4: GELİŞTİRİCİ EKİP
# ---------------------------------------------------------
with sekme_ekip:
    st.markdown('<div class="custom-header">Proje Ekibi</div>', unsafe_allow_html=True)
    st.write("Bu kısma ekip üyelerinin isimlerini ve iletişim bilgilerini ekleyebilirsiniz.")