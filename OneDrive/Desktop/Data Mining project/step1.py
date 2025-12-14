import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. DATA ACQUISITION
print("Fetching data...")
ticker = "GBPTRY=X"
df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)

# --- KRİTİK DÜZELTME (FIX) ---
# Yeni yfinance sürümü 'MultiIndex' (iç içe sütun) döndürüyor.
# Bunu düzeltip sadece 'Close', 'Open' gibi tek başlığa indiriyoruz.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
# -----------------------------

# 'Adj Close' is generally for stock splits, 'Close' is sufficient for forex.
df = df[['Close']].copy()

# 2. DATA CLEANING
if df.isnull().values.any():
    df = df.fillna(method='ffill')

# 3. FEATURE ENGINEERING
# a) Daily Return
df['Return'] = df['Close'].pct_change()

# b) Moving Averages
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# c) Volatility
df['Volatility'] = df['Close'].rolling(window=20).std()

# d) Labeling (Target for Classification)
# Shift işlemini yaparken seri (Series) olduğundan emin oluyoruz.
df['Tomorrow_Close'] = df['Close'].shift(-1)

# Hatanın çıktığı yer burasıydı. Artık sütunlar düzgün olduğu için hata vermeyecek.
# (df['Tomorrow_Close'] > df['Close']) karşılaştırması artık güvenli.
df['Target_Class'] = (df['Tomorrow_Close'] > df['Close']).astype(int)

# Drop NaNs
df.dropna(inplace=True)

# 4. STATISTICS
print("-" * 30)
print("DATA STATISTICS")
print("-" * 30)
print(df.describe())

# 5. SAVING DATA
df.to_csv('gbp_project_data.csv')
print("\nData cleaned and saved as 'gbp_project_data.csv'.")

# 6. VISUALIZATION
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='GBP/TRY Rate')
plt.plot(df.index, df['SMA_50'], label='50-Day SMA', color='orange', alpha=0.7)
plt.legend()
plt.title('GBP/TRY Time Series Plot')
plt.xlabel('Year')
plt.ylabel('Price (TRY)')
plt.show()