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
# FEATURE ENGINEERING (İndikatörler)
# ======================================================
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['Close'])

exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

df.dropna(inplace=True)

# ÖNEMLİ DEĞİŞİKLİK: Hedef Değişkeni "Fiyat Farkı" yapıyoruz.
# Model "Yarın 45 TL olacak" demeyecek, "Yarın bugüne göre +0.20 TL artacak" diyecek.
df['Target_Diff'] = df['Tomorrow_Close'] - df['Close']

features = ['Close', 'SMA_50', 'SMA_200', 'Volatility', 'Return', 'RSI', 'MACD', 'Signal_Line']

X = df[features]
y_classification = df['Target_Class']
y_forecasting_diff = df['Target_Diff'] # Fiyat yerine FARK'ı öğretiyoruz.

# 2. DATA SPLIT (Time-Based)
split_point = int(len(df) * 0.8)

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]

# Classification Targets
y_class_train = y_classification.iloc[:split_point]
y_class_test = y_classification.iloc[split_point:]

# Forecasting Targets (Diff)
y_forecast_train = y_forecasting_diff.iloc[:split_point]
y_forecast_test_diff = y_forecasting_diff.iloc[split_point:]

# Gerçek Fiyatları Saklayalım (Test aşamasında karşılaştırmak için)
actual_prices_test = df['Tomorrow_Close'].iloc[split_point:]
current_prices_test = df['Close'].iloc[split_point:]

print(f"Eğitim: {len(X_train)} gün, Test: {len(X_test)} gün.")

# ==========================================
# PART A: CLASSIFICATION
# ==========================================
print("\n--- Sınıflandırma Modeli ---")
clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
clf.fit(X_train, y_class_train)
y_class_pred = clf.predict(X_test)

cm = confusion_matrix(y_class_test, y_class_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Classification Matrix')
plt.savefig('grafik1_final_classification.png')
print(">> Grafik 1 Kaydedildi.")

# ==========================================
# PART B: FORECASTING (KRİTİK DÜZELTME)
# ==========================================
print("\n--- Tahminleme Modeli (Fark Tahmini) ---")

reg = RandomForestRegressor(n_estimators=200, random_state=42)
# Modeli "Fiyat Farkı" üzerine eğitiyoruz
reg.fit(X_train, y_forecast_train)

# Model "Yarın ne kadar artacak?" sorusuna cevap veriyor
predicted_diff = reg.predict(X_test)

# TAHMİNİ FİYATA DÖNÜŞTÜRME
# Yarınki Fiyat = Bugünün Fiyatı + Tahmin Edilen Artış Miktarı
predicted_prices = current_prices_test + predicted_diff

# Hata Hesaplama
rmse = np.sqrt(mean_squared_error(actual_prices_test, predicted_prices))
print(f"Hata Oranı (RMSE): {rmse:.4f} TL")

# Grafiği Çiz ve Kaydet
plt.figure(figsize=(14,7))
plt.plot(actual_prices_test.index, actual_prices_test.values, label='Gerçek Fiyat', color='blue', linewidth=2)
plt.plot(actual_prices_test.index, predicted_prices, label='Tahmin Edilen (AI)', color='red', linestyle='--', linewidth=2)
plt.title('GBP/TRY Fiyat Tahmini (Düzeltilmiş - Difference Prediction)')
plt.xlabel('Zaman')
plt.ylabel('Fiyat (TL)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafik2_final_forecasting.png')
print(">> Grafik 2 Kaydedildi: Artık kırmızı çizgi maviye yakın olmalı!")

# ==========================================
# PART C: FEATURE IMPORTANCE
# ==========================================
importances = reg.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Model Fiyatı Tahmin Ederken Neye Baktı?")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('grafik3_final_importance.png')
print(">> Grafik 3 Kaydedildi.")