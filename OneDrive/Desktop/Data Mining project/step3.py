import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# 1. VERİYİ YÜKLEME
print("Veri yükleniyor...")
df = pd.read_csv('gbp_project_data.csv', index_col=0)

# ==========================================
# EKSİK PARÇA: RSI HESAPLAMA (Fix)
# ==========================================
# Bu fonksiyonu buraya ekledik ki 'KeyError: RSI' hatası almayalım.
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# RSI'ı tekrar hesaplayıp tabloya ekliyoruz
df['RSI'] = calculate_rsi(df['Close'])

# Hesaplama yüzünden oluşan ilk boş satırları silelim (Yoksa hata verir)
df.dropna(inplace=True)

# ==========================================
# BÖLÜM A: CLUSTERING (WEKA GÖRÜNÜMLÜ)
# ==========================================
print("\n=== CLUSTERING (K-MEANS) SONUÇLARI ===")

# Kümeleme için kullanılacak özellikler
features = ['Return', 'Volatility', 'RSI']
X_cluster = df[features].copy()

# Standardizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-Means (3 Küme)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# --- WEKA TARZI TEXT ÇIKTISI (CENTROIDS) ---
centroids = df.groupby('Cluster')[features].mean()
counts = df['Cluster'].value_counts().sort_index()

print("\nCluster Centroids (Küme Merkezleri):")
print("-" * 65)
print(f"{'Attribute':<15} {'Cluster 0':<15} {'Cluster 1':<15} {'Cluster 2':<15}")
print("-" * 65)

for feature in features:
    row = f"{feature:<15} "
    for i in range(3):
        # Eğer küme numarası varsa değeri al, yoksa 0 yaz (Hata önleyici)
        if i in centroids.index:
            val = centroids.loc[i, feature]
            row += f"{val:<15.4f} "
        else:
            row += f"{0:<15.4f} "
    print(row)
print("-" * 65)

# Yüzdeleri hesapla
total_len = len(df)
p0 = (counts.get(0, 0) / total_len * 100)
p1 = (counts.get(1, 0) / total_len * 100)
p2 = (counts.get(2, 0) / total_len * 100)

print(f"{'INSTANCES':<15} {counts.get(0,0):<15} {counts.get(1,0):<15} {counts.get(2,0):<15}")
print(f"{'PERCENTAGE':<15} %{p0:.0f}            %{p1:.0f}            %{p2:.0f}")
print("-" * 65)

# Kümeleri Anlamlandıralım
vol_means = df.groupby('Cluster')['Volatility'].mean().sort_values()
labels = {
    vol_means.index[0]: 'Sakin (Low Risk)',
    vol_means.index[1]: 'Orta (Medium)',
    vol_means.index[2]: 'Kriz (High Risk)'
}
df['Cluster_Label'] = df['Cluster'].map(labels)

# --- GRAFİK (SCATTER PLOT) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Return', y='Volatility', hue='Cluster_Label', palette='bright', s=60, alpha=0.8)

# Merkezleri çiz
centers = df.groupby('Cluster_Label')[['Return', 'Volatility']].mean()
plt.scatter(centers['Return'], centers['Volatility'], c='black', s=200, marker='X', label='Centroids')

plt.title('Weka Style Clustering Result')
plt.xlabel('Return')
plt.ylabel('Volatility')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('grafik4_weka_clustering.png')
print("\n>> Grafik 4 Kaydedildi: grafik4_weka_clustering.png")

# ==========================================
# BÖLÜM B: ASSOCIATION RULES (WEKA TARZI)
# ==========================================
print("\n=== ASSOCIATION RULES (APRIORI) SONUÇLARI ===")

df_assoc = pd.DataFrame()
# Boolean (True/False) dönüşümü yapıyoruz
df_assoc['Price=UP'] = df['Return'] > 0 
df_assoc['Vol=HIGH'] = df['Volatility'] > df['Volatility'].mean()
df_assoc['Trend=UP'] = df['Close'] > df['SMA_50']
df_assoc['RSI=HIGH'] = df['RSI'] > 70

frequent_itemsets = apriori(df_assoc, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# En güçlü 10 kural
rules = rules.sort_values('confidence', ascending=False).head(10)

print("\nBest Rules found:")
for index, row in rules.iterrows():
    ant = list(row['antecedents'])[0]
    con = list(row['consequents'])[0]
    conf = row['confidence']
    print(f"{index+1}. {ant} ==> {con}   conf:({conf:.2f})")

# Kuralları kaydet
rules.to_csv('step3.csv')
print("\n>> Kurallar kaydedildi: step3_weka_rules.csv")