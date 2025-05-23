import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os

# Load data
df = pd.read_csv("tmdb_5000_movies.csv")

# Ambil fitur yang dibutuhkan
df['main_genre'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')
genre_dummies = pd.get_dummies(df['main_genre'])

# Pastikan kolom genre konsisten, tambahkan genre yang mungkin tidak ada di subset data
all_genres = ['Action', 'Comedy', 'Drama', 'Horror'] # Sesuaikan dengan semua genre yang relevan
for genre in all_genres:
    if genre not in genre_dummies.columns:
        genre_dummies[genre] = 0

# Pilih fitur dan pastikan urutannya konsisten
feature_columns = ['vote_average', 'vote_count'] + all_genres
features = pd.concat([df[feature_columns[:2]], genre_dummies[all_genres]], axis=1)

# Tangani nilai NaN jika ada (meskipun scaler handle NaN, preprocessing mungkin butuh)
features.fillna(0, inplace=True)

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# KMeans Clustering dengan K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # Tambahkan n_init
kmeans.fit(X_scaled)

# Buat direktori model jika belum ada
if not os.path.exists('model'):
    os.makedirs('model')

# Simpan model dan scaler
with open('model/kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model K-Means dan Scaler berhasil dilatih dan disimpan di direktori 'model/'.") 