import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Kripto AI Uzmanı", layout="wide")

st.title("🚀 Kripto Sinyal ve AI Tahmin Paneli")
st.write("Bu panel, teknik analiz (RSI) ve Yapay Zeka (Regresyon) kullanarak piyasayı yorumlar.")

# --- YAN MENÜ ---
st.sidebar.header("Ayarlar")
secilen_coinler = st.sidebar.multiselect(
    "Analiz Edilecek Coinler", 
    ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "AVAX-USD", "DOGE-USD"],
    default=["BTC-USD", "ETH-USD", "SOL-USD"]
)
rsi_periyot = st.sidebar.slider("RSI Hassasiyeti", 5, 30, 14)
tahmin_gun = st.sidebar.slider("Gelecek Tahmini (Gün)", 1, 30, 7) # Kaç gün sonrasını tahmin etsin?
taramayi_baslat = st.sidebar.button("Analizi Başlat")

# --- FONKSİYONLAR ---
def rsi_hesapla(veri, periyot=14):
    delta = veri['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=periyot).mean()
    avg_loss = loss.rolling(window=periyot).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- YAPAY ZEKA TAHMİN FONKSİYONU ---
def gelecek_tahmini_yap(veri, gun_sayisi):
    # Veriyi hazırlama
    veri = veri.reset_index()
    veri['Gun_No'] = veri.index
    
    X = veri[['Gun_No']] # Girdi: Gün numarası
    y = veri['Close']    # Çıktı: Fiyat
    
    # Modeli Eğit (Linear Regression)
    model = LinearRegression()
    model.fit(X, y)
    
    # Gelecek günleri oluştur
    son_gun_no = veri['Gun_No'].iloc[-1]
    gelecek_gunler = np.array([[son_gun_no + i] for i in range(1, gun_sayisi + 1)])
    
    # Tahmin yap
    tahminler = model.predict(gelecek_gunler)
    
    return gelecek_gunler, tahminler, model

# --- ANA İŞLEM ---
if taramayi_baslat:
    col1, col2 = st.columns([1, 2]) # Ekranı ikiye böl (Sol: Tablo, Sağ: Grafik)
    
    rapor_listesi = []
    
    # 1. ANALİZ KISMI
    for symbol in secilen_coinler:
        try:
            veri = yf.download(symbol, period="6mo", interval="1d", progress=False)
            if isinstance(veri.columns, pd.MultiIndex):
                veri.columns = veri.columns.droplevel(1)
            
            veri['RSI'] = rsi_hesapla(veri, rsi_periyot)
            
            son_fiyat = float(veri['Close'].iloc[-1])
            son_rsi = float(veri['RSI'].iloc[-1])
            
            durum = "NÖTR"
            if son_rsi < 30: durum = "🟢 AL FIRSATI"
            elif son_rsi > 70: durum = "🔴 SATIŞ RİSKİ"
            
            rapor_listesi.append({
                "Coin": symbol,
                "Fiyat ($)": f"{son_fiyat:.2f}",
                "RSI": f"{son_rsi:.2f}",
                "Sinyal": durum
            })
            
        except Exception as e:
            st.error(f"{symbol} hatası.")

    # Sonuçları Sol Tarafa Yaz
    with col1:
        st.subheader("📋 Piyasa Durumu")
        st.dataframe(pd.DataFrame(rapor_listesi))

    # 2. YAPAY ZEKA GRAFİK KISMI
    with col2:
        if len(secilen_coinler) > 0:
            coin = secilen_coinler[0] # İlk seçilen coini grafiğe dök
            st.subheader(f"🤖 {coin} - Yapay Zeka Tahmini")
            
            # Veriyi tekrar çek (Grafik için)
            veri_ai = yf.download(coin, period="6mo", interval="1d", progress=False)
            if isinstance(veri_ai.columns, pd.MultiIndex):
                veri_ai.columns = veri_ai.columns.droplevel(1)
            
            # AI Modeli Çalıştır
            gelecek_x, gelecek_y, model = gelecek_tahmini_yap(veri_ai, tahmin_gun)
            
            # Grafiği Hazırla
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 5))
            # Geçmiş Fiyatlar
            ax.plot(veri_ai.index, veri_ai['Close'], label="Gerçek Fiyat", color="blue")
            
            # Trend Çizgisi (Regresyon Doğrusu)
            tum_tahmin = model.predict(np.array(range(len(veri_ai))).reshape(-1, 1))
            ax.plot(veri_ai.index, tum_tahmin, label="Genel Trend (AI)", color="orange", linestyle="--", alpha=0.7)
            
            # Gelecek Tahmini (Kırmızı Noktalar)
            # Tarihleri oluştur
            son_tarih = veri_ai.index[-1]
            gelecek_tarihler = [son_tarih + pd.Timedelta(days=i) for i in range(1, tahmin_gun + 1)]
            
            ax.plot(gelecek_tarihler, gelecek_y, label=f"Gelecek {tahmin_gun} Gün Tahmini", color="red", marker="o", linestyle="-")
            
            ax.set_title(f"{coin} Fiyat Tahmin Modeli")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Tahmin Yorumu
            egim = model.coef_[0]
            if egim > 0:
                st.success(f"Yapay Zeka Yorumu: {coin} genel trendi **YUKARI** yönlü. 🚀")
            else:
                st.warning(f"Yapay Zeka Yorumu: {coin} genel trendi **AŞAĞI** yönlü. 🔻")

else:
    st.info("👈 Analizi başlatmak için soldaki butona basın.")