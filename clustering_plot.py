import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Membuat data sampel yang menyerupai plot
# Data ini hanya ilustrasi dan mungkin perlu diganti dengan data sebenarnya
data = {
    'IPK': np.random.uniform(2.0, 4.0, 50),
    'Jumlah SKS': np.random.uniform(40, 150, 50),
    'Cluster': np.random.randint(0, 3, 50)
}

df = pd.DataFrame(data)

# Load data
df = pd.read_csv("tmdb_5000_movies.csv")

# Ambil fitur yang dibutuhkan
df['main_genre'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')
genre_dummies = pd.get_dummies(df['main_genre'])

features = pd.concat([df[['vote_average', 'vote_count']], genre_dummies], axis=1)

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# KMeans Clustering dengan K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # Tambahkan n_init untuk menghindari warning
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Membuat plot scatter menggunakan seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='vote_average', y='vote_count', hue='Cluster', palette='viridis', s=50)

# Mengatur judul dan label sumbu
plt.title('Hasil Clustering Film Berdasarkan Rating dan Jumlah Vote', fontsize=16)
plt.xlabel('Rata-rata Vote', fontsize=12)
plt.ylabel('Jumlah Vote', fontsize=12)

# Menampilkan plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show() 