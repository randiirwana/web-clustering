import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("tmdb_5000_movies.csv")

# Ambil fitur yang dibutuhkan
df['main_genre'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')
genre_dummies = pd.get_dummies(df['main_genre'])

features = pd.concat([df[['vote_average', 'vote_count']], genre_dummies], axis=1)

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Inertia")
plt.show()

# KMeans Clustering dengan K=4
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Tampilkan hasil clustering
print(df[['title', 'vote_average', 'vote_count', 'main_genre', 'Cluster']].head())
