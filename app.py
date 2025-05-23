from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import matplotlib
matplotlib.use('Agg')  # Set backend ke Agg sebelum mengimpor pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import time
from sklearn.metrics import silhouette_score

app = Flask(__name__)

# Lokasi model dan scaler yang sudah dilatih
MODEL_PATH = 'model/kmeans_model.pkl'
SCALER_PATH = 'model/scaler.pkl'

# Load model dan scaler saat aplikasi dimulai
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print("Model atau scaler tidak ditemukan. Harap jalankan train_model.py terlebih dahulu.")

def get_plot_base64():
    # Buat plot di memori
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()  # Pastikan untuk menutup plot
    # Encode ke base64
    graph = base64.b64encode(image_png).decode('utf-8')
    return graph

@app.route('/')
def home():
    if not model_loaded:
        return render_template('index.html', error="Model clustering belum dilatih. Harap jalankan train_model.py.")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if not model_loaded:
        return render_template('index.html', error="Model clustering belum dilatih. Harap jalankan train_model.py.")

    if 'dataset' not in request.files:
        return render_template('index.html', error="Tidak ada file dataset yang diunggah.")

    file = request.files['dataset']

    if file.filename == '':
        return render_template('index.html', error="Nama file kosong.")

    if file and file.filename.endswith('.csv'):
        try:
            # Baca file CSV
            df = pd.read_csv(file)

            # --- Preprocessing dan Clustering (sesuai dengan train_model.py) ---
            if 'genres' in df.columns:
                df['main_genre'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if pd.notna(x) and x and eval(x) else 'Unknown')
            else:
                 df['main_genre'] = 'Unknown'

            genre_dummies = pd.get_dummies(df['main_genre'])

            all_genres = ['Action', 'Comedy', 'Drama', 'Horror']
            for genre in all_genres:
                if genre not in genre_dummies.columns:
                    genre_dummies[genre] = 0

            feature_columns = ['vote_average', 'vote_count'] + all_genres
            
            selected_features_df = pd.DataFrame(index=df.index)
            if 'vote_average' in df.columns: selected_features_df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
            if 'vote_count' in df.columns: selected_features_df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')

            selected_genre_dummies = genre_dummies[all_genres] if all(item in genre_dummies.columns for item in all_genres) else pd.DataFrame(0, index=df.index, columns=all_genres)

            features = pd.concat([selected_features_df, selected_genre_dummies], axis=1)
            features.fillna(0, inplace=True)
            
            if features.empty or features.isnull().all().all():
                 return render_template('index.html', error="Data setelah preprocessing kosong atau tidak valid.")

            print("Jumlah data setelah preprocessing:", len(df))

            print(df[['vote_average', 'vote_count']].head())
            print(df[['vote_average', 'vote_count']].dtypes)

            X_scaled = scaler.transform(features)
            clusters = model.predict(X_scaled)
            df['Cluster'] = clusters

            # --- Visualisasi ---
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='vote_average', y='vote_count', hue='Cluster', palette='viridis', s=50)
            plt.title('Hasil Clustering Film Berdasarkan Rating dan Jumlah Vote')
            plt.xlabel('Rata-rata Vote')
            plt.ylabel('Jumlah Vote')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Konversi plot ke base64
            plot_base64 = get_plot_base64()
            plt.close()

            display_columns = ['budget', 'homepage', 'vote_average', 'vote_count', 'main_genre', 'Cluster']
            df[display_columns] = df[display_columns].fillna('-')
            headers = [h.replace('_', ' ').title() for h in display_columns]
            results = df[display_columns].values.tolist()

            print("Jumlah baris yang dikirim ke template:", len(results))
            print("Contoh baris results:", results[0])

            print("Headers:", headers)
            print("Contoh baris results:", results[0])

            return render_template('index.html', headers=headers, results=results, plot_base64=plot_base64)

        except Exception as e:
            return render_template('index.html', error=f"Terjadi kesalahan saat memproses file: {e}")

    else:
        return render_template('index.html', error="Format file tidak didukung. Mohon unggah file CSV.")

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return render_template('index.html', error="Model clustering belum dilatih. Harap jalankan train_model.py.")

    try:
        vote_avg = float(str(request.form['rating']).replace(',', '.'))
        vote_cnt = float(str(request.form['votes']).replace(',', '.'))
        genre_input = request.form['genre']

        genre_list = ['Action', 'Comedy', 'Drama', 'Horror']
        genre_encoding = [1 if g == genre_input else 0 for g in genre_list]
        input_data = [vote_avg, vote_cnt] + genre_encoding
        input_data_np = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_data_np)
        cluster = model.predict(input_scaled)[0]

        df = pd.read_csv("tmdb_5000_movies.csv")
        df['main_genre'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')
        genre_dummies = pd.get_dummies(df['main_genre'])
        for genre in genre_list:
            if genre not in genre_dummies.columns:
                genre_dummies[genre] = 0
        features = pd.concat([df[['vote_average', 'vote_count']], genre_dummies[genre_list]], axis=1)
        features.fillna(0, inplace=True)
        X_scaled = scaler.transform(features)
        clusters_all = model.predict(X_scaled)
        df['Cluster'] = clusters_all

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='vote_average', y='vote_count', hue='Cluster', palette='viridis', s=50, alpha=0.6)
        plt.scatter([vote_avg], [vote_cnt], color='red', s=120, label='Data Input', edgecolor='black', zorder=10)
        plt.title('Hasil Clustering & Data yang Diprediksi')
        plt.xlabel('Rata-rata Vote')
        plt.ylabel('Jumlah Vote')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Konversi plot ke base64
        plot_base64 = get_plot_base64()
        plt.close()

        return render_template(
            'index.html',
            result=f"Film termasuk ke dalam cluster: {cluster}",
            rating=vote_avg,
            votes=vote_cnt,
            genre=genre_input,
            plot_base64=plot_base64
        )

    except ValueError:
        return render_template('index.html', error="Input tidak valid. Pastikan rating dan jumlah vote adalah angka.")
    except Exception as e:
        return render_template('index.html', error=f"Terjadi kesalahan saat prediksi: {e}")

@app.route('/elbow')
def elbow():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df['main_genre'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')
    genre_dummies = pd.get_dummies(df['main_genre'])
    all_genres = ['Action', 'Comedy', 'Drama', 'Horror']
    for genre in all_genres:
        if genre not in genre_dummies.columns:
            genre_dummies[genre] = 0
    features = pd.concat([df[['vote_average', 'vote_count']], genre_dummies[all_genres]], axis=1)
    features.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(K_range, inertia, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Inertia")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Konversi plot ke base64
    plot_base64 = get_plot_base64()
    plt.close()

    return render_template('elbow.html', plot_base64=plot_base64)

@app.route('/elbow_data')
def elbow_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df['main_genre'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')
    genre_dummies = pd.get_dummies(df['main_genre'])
    all_genres = ['Action', 'Comedy', 'Drama', 'Horror']
    for genre in all_genres:
        if genre not in genre_dummies.columns:
            genre_dummies[genre] = 0
    features = pd.concat([df[['vote_average', 'vote_count']], genre_dummies[all_genres]], axis=1)
    features.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(K_range, inertia, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Inertia")
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_base64 = get_plot_base64()
    # plt.close() sudah dipanggil di get_plot_base64
    return jsonify({'plot_base64': plot_base64})

@app.route('/silhouette_data')
def silhouette_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df['main_genre'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')
    genre_dummies = pd.get_dummies(df['main_genre'])
    all_genres = ['Action', 'Comedy', 'Drama', 'Horror']
    for genre in all_genres:
        if genre not in genre_dummies.columns:
            genre_dummies[genre] = 0
    features = pd.concat([df[['vote_average', 'vote_count']], genre_dummies[all_genres]], axis=1)
    features.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 6))
    plt.plot(list(K_range), silhouette_scores, marker='o')
    plt.title("Metode Silhouette Score")
    plt.xlabel("Jumlah Klaster (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_base64 = get_plot_base64()
    return jsonify({'plot_base64': plot_base64})

if __name__ == '__main__':
    app.run(debug=True)
