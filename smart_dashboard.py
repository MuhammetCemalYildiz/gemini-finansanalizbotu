import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

# .env dosyasindan verileri yukle
load_dotenv()
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Smart Crypto AI Dashboard", layout="wide")

# Custom CSS for a more premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Smart Crypto Sinyal & AI Asistanı")
st.write("Bu panel, teknik analiz, lineer regresyon ve **Google Gemini AI** kullanarak piyasayı derinlemesine yorumlar.")

# --- YAN MENÜ ---
st.sidebar.header("⚙️ Ayarlar")

# Otomatik Konfigurasyon (UI'dan kaldirildi)
api_key = DEFAULT_API_KEY
gemini_model_adi = "models/gemini-2.5-flash"

if not api_key:
    st.error("⚠️ .env dosyasinda API anahtari bulunamadi! Lutfen 'baslat.bat' klasorundeki .env dosyasina anahtarinizi ekleyin.")
secilen_coinler = st.sidebar.multiselect(
    "Analiz Edilecek Coinler", 
    ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "LINK-USD", "ADA-USD"],
    default=["BTC-USD", "ETH-USD"]
)

rsi_periyot = st.sidebar.slider("RSI Periyodu", 5, 30, 14)
tahmin_gun = st.sidebar.slider("Gelecek Tahmini (Gün)", 1, 30, 7)
taramayi_baslat = st.sidebar.button("📊 Analizi Başlat", use_container_width=True)

# --- FONKSİYONLAR ---
def rsi_hesapla(veri, periyot=14):
    delta = veri['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=periyot).mean()
    avg_loss = loss.rolling(window=periyot).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def gelecek_tahmini_yap(veri, gun_sayisi):
    veri = veri.reset_index()
    veri['Gun_No'] = veri.index
    X = veri[['Gun_No']]
    y = veri['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    son_gun_no = veri['Gun_No'].iloc[-1]
    gelecek_gunler = np.array([[son_gun_no + i] for i in range(1, gun_sayisi + 1)])
    tahminler = model.predict(gelecek_gunler)
    
    return gelecek_gunler, tahminler, model

def gemini_yorumu_al(api_key, coin, fiyat, rsi, trend, model_name):
    if not api_key:
        return "⚠️ Gemini API anahtarı girilmediği için detaylı yorum yapılamıyor. Lütfen sidebar'a anahtarınızı girin."
    
    # API Anahtarını temizle (boşluk vs. temizliği gRPC hatalarını önler)
    api_key = api_key.strip()
    
    try:
        genai.configure(api_key=api_key, transport='rest')
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        Sen bir kripto para uzmanısın. Şu anki veriler şöyle:
        Coin: {coin}
        Son Fiyat: {fiyat} USD
        RSI Değeri (14): {rsi:.2f}
        Genel Trend (Regresyon): {trend}
        
        Bu verilere dayanarak COK KISA ve NET bir piyasa yorumu yap (Maksimum 3 cumle). 
        Yatirim tavsiyesi olmadigini belirtme (zaten biliyoruz), sadece teknik ve psikolojik durumu ozetle.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        hata_mesaji = str(e)
        if "404" in hata_mesaji:
            return "❌ MODEL BULUNAMADI: Secilen model hesabinizda aktif degil. Lutfen Sidebar'daki 'API Test' butonuna basip GUNCEL listeden bir model secin."
        return f"Gemini Hatasi: {hata_mesaji}"

# --- ANA İŞLEM ---
if taramayi_baslat:
    # 1. ANALİZ VE METRİKLER
    st.subheader("📍 Anlık Piyasa Özeti")
    metrik_cols = st.columns(len(secilen_coinler))
    
    all_data = {}
    
    for i, symbol in enumerate(secilen_coinler):
        try:
            veri = yf.download(symbol, period="6mo", interval="1d", progress=False)
            if isinstance(veri.columns, pd.MultiIndex):
                veri.columns = veri.columns.droplevel(1)
            
            veri['RSI'] = rsi_hesapla(veri, rsi_periyot)
            all_data[symbol] = veri
            
            son_fiyat = float(veri['Close'].iloc[-1])
            onceki_fiyat = float(veri['Close'].iloc[-2])
            degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
            
            with metrik_cols[i]:
                st.metric(label=symbol, value=f"${son_fiyat:,.2f}", delta=f"{degisim:.2f}%")
        except Exception as e:
            st.error(f"{symbol} verisi alınamadı.")

    st.divider()

    # 2. DETAYLI ANALİZ VE GRAFİKLER
    for symbol in secilen_coinler:
        try:
            if symbol in all_data:
                veri = all_data[symbol]
                son_rsi = float(veri['RSI'].iloc[-1])
                
                with st.expander(f"🔍 {symbol} Detaylı Analiz & AI Yorumu", expanded=True):
                    col_chart, col_ai = st.columns([2, 1])
                    
                    with col_chart:
                        # AI Modeli Çalıştır
                        gelecek_x, gelecek_y, model = gelecek_tahmini_yap(veri, tahmin_gun)
                        egim = model.coef_[0]
                        trend_metni = "YUKARI" if egim > 0 else "AŞAĞI"
                        
                        # Grafiği Hazırla
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
                        plt.subplots_adjust(hspace=0.1)
                        
                        # Fiyat Grafiği
                        ax1.plot(veri.index, veri['Close'], label="Gerçek Fiyat", color="#1c92d2", linewidth=2)
                        
                        # Gelecek Tahmini
                        son_tarih = veri.index[-1]
                        gelecek_tarihler = [son_tarih + pd.Timedelta(days=i) for i in range(1, tahmin_gun + 1)]
                        ax1.plot(gelecek_tarihler, gelecek_y, label="AI Tahmini", color="#f06292", marker="o", linestyle="--")
                        
                        ax1.set_ylabel("Fiyat ($)")
                        ax1.legend()
                        ax1.grid(True, alpha=0.2)
                        ax1.set_title(f"{symbol} Fiyat ve Tahmin Modeli", fontsize=14)
                        
                        # RSI Grafiği
                        ax2.plot(veri.index, veri['RSI'], color="#ffca28", label="RSI")
                        ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
                        ax2.axhline(30, color="green", linestyle="--", alpha=0.5)
                        ax2.fill_between(veri.index, 70, 100, color="red", alpha=0.1)
                        ax2.fill_between(veri.index, 0, 30, color="green", alpha=0.1)
                        ax2.set_ylabel("RSI")
                        ax2.set_ylim(0, 100)
                        
                        st.pyplot(fig)
                    
                    with col_ai:
                        st.write("### 🤖 Gemini AI Analizi")
                        # Rate limit (Ücretsiz katman) için kısa bir bekleme ekleyelim
                        if len(secilen_coinler) > 1:
                            time.sleep(1) 
                        
                        print(f"DEBUG: {symbol} için AI yorumu isteniyor... (Model: {gemini_model_adi})")
                        insight = gemini_yorumu_al(api_key, symbol, float(veri['Close'].iloc[-1]), son_rsi, trend_metni, gemini_model_adi)
                        
                        # Ciktiyi kaydirilabilir kutu icine al
                        with st.container(height=150):
                            st.info(insight)
                        
                        st.write("### 📊 Teknik Durum")
                        if son_rsi < 30:
                            st.success("🟢 **AŞIRI SATIM (ALIM FIRSATI)**: RSI 30'un altında.")
                        elif son_rsi > 70:
                            st.error("🔴 **AŞIRI ALIM (SATIŞ RİSKİ)**: RSI 70'in üzerinde.")
                        else:
                            st.warning("⚪ **NÖTR BÖLGE**: Karar vermek için beklemek gerekebilir.")
                        
                        st.write(f"**Genel Trend:** {'📈 YUKARI' if egim > 0 else '📉 AŞAĞI'}")
        except Exception as e:
            st.error(f"{symbol} analizi sırasında bir hata oluştu: {str(e)}")
            print(f"HATA: {symbol} analiz hatası -> {str(e)}")

else:
    # Karşılama Ekranı
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1621761191319-c6fb62004040?q=80&w=1000&auto=format&fit=crop", caption="Akıllı Finansal Analiz Paneli")
        st.info("👈 Analizi başlatmak için soldaki ayarları yapın ve **'Analizi Başlat'** butonuna basın.")
