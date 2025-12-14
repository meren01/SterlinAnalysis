import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
import numpy as np

# 1. VERİYİ YÜKLE
print("Veri yükleniyor...")
df = pd.read_csv('gbp_project_data.csv', index_col=0)

# ======================================================
# İYİLEŞTİRME 1: YENİ ÖZNİTELİKLER (FEATURE ENGINEERING)
# ======================================================
# RSI ve MACD hesaplayarak modele "Borsa Uzmanı" yetenekleri ekliyoruz.

# RSI Hesaplama Fonksiyonu
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['Close'])

# MACD Hesaplama
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Yeni oluşan NaN değerleri temizle (indikatör hesaplarken oluşur)
df.dropna(inplace=True)

# Yeni Özellik Listemiz (Daha Zengin)
features = ['Close', 'SMA_50', 'SMA_200', 'Volatility', 'Return', 'RSI', 'MACD', 'Signal_Line']

X = df[features]
y_classification = df['Target_Class']
y_forecasting = df['Tomorrow_Close']

# 2. DATA SPLIT (Time-Based)
split_point = int(len(df) * 0.8)

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]

y_class_train = y_classification.iloc[:split_point]
y_class_test = y_classification.iloc[split_point:]

y_forecast_train = y_forecasting.iloc[:split_point]
y_forecast_test = y_forecasting.iloc[split_point:]

print(f"Eğitim Verisi: {len(X_train)} gün, Test Verisi: {len(X_test)} gün.")

# ==========================================
# PART A: CLASSIFICATION (Daha Dengeli)
# ==========================================
print("\n--- Sınıflandırma Modeli Çalışıyor ---")

# İYİLEŞTİRME 2: class_weight='balanced'
# Bu parametre modelin "Sürekli Düşecek" demesini engeller, azınlık sınıfa odaklanır.
clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
clf.fit(X_train, y_class_train)

y_class_pred = clf.predict(X_test)

# Sonuçlar
acc = accuracy_score(y_class_test, y_class_pred)
cm = confusion_matrix(y_class_test, y_class_pred)
print(f"Yeni Accuracy: %{acc*100:.2f}")
print("\nDetaylı Rapor:\n", classification_report(y_class_test, y_class_pred))

# Grafiği Kaydet
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title('Gelişmiş Classification Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.tight_layout()
plt.savefig('grafik1_classification_matrix.png') # DOSYA OLARAK KAYDEDER
print(">> Grafik 1 kaydedildi: grafik1_classification_matrix.png")
# plt.show() # Kodu durdurmasın diye yorum satırı yaptım, dosya olarak bakabilirsin.

# ==========================================
# PART B: FORECASTING (Fiyat Tahmini)
# ==========================================
print("\n--- Tahminleme Modeli Çalışıyor ---")

reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
reg.fit(X_train, y_forecast_train)

y_forecast_pred = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_forecast_test, y_forecast_pred))
print(f"Hata Oranı (RMSE): {rmse:.4f} TL")

# Grafiği Kaydet
plt.figure(figsize=(14,7))
plt.plot(y_forecast_test.index, y_forecast_test.values, label='Gerçek Fiyat', color='blue', linewidth=2)
plt.plot(y_forecast_test.index, y_forecast_pred, label='Tahmin Edilen (AI)', color='red', linestyle='--', linewidth=2)
plt.title('GBP/TRY Fiyat Tahmini (Geliştirilmiş Model)')
plt.xlabel('Test Günleri')
plt.ylabel('Fiyat (TL)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafik2_forecasting.png') # DOSYA OLARAK KAYDEDER
print(">> Grafik 2 kaydedildi: grafik2_forecasting.png")

# ==========================================
# PART C: FEATURE IMPORTANCE (Hangi Veri Önemli?)
# ==========================================
importances = reg.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Model Hangi Veriye Bakarak Karar Verdi?")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='purple')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('grafik3_feature_importance.png')
print(">> Grafik 3 kaydedildi: grafik3_feature_importance.png")

print("\nBÜTÜN İŞLEMLER TAMAMLANDI. Masaüstündeki PNG dosyalarını kontrol et.")