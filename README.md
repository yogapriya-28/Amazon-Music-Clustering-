# 🎧 Amazon Music Clustering

Unsupervised Machine Learning project that clusters Amazon Music songs based on their **audio features** — uncovering natural groupings of tracks by mood, energy, instrumentation, and more.

---

## 📌 Project Overview

- **Domain**: Music Analytics / Unsupervised Learning  
- **Goal**: Automatically cluster songs based on audio characteristics to infer potential genres, moods, or playlist themes — with **no labels** required.

---

## 🔍 Problem Statement

With millions of tracks on streaming platforms, **manual genre classification** is inefficient. This project uses **clustering algorithms** to:
- Discover hidden structure in audio features
- Enable better recommendations and playlists
- Help artists and platforms analyze track similarities

---

## 💡 Business Use Cases

| Use Case               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| 🎵 Playlist Curation   | Group similar songs for **automatic playlist generation**                   |
| 🔍 Music Discovery     | Recommend tracks with a similar audio profile                               |
| 👩‍🎤 Artist Insights    | Understand competitors and similar-sounding songs                           |
| 📊 Market Segmentation | Analyze listener preferences by sound clusters                              |

---

## 🛠️ Skills & Tools Used

- **Data Science**: EDA, preprocessing, normalization, clustering, dimensionality reduction  
- **ML Algorithms**: K-Means, DBSCAN, Hierarchical Clustering, PCA  
- **Libraries**: `pandas`, `NumPy`, `scikit-learn`, `matplotlib`, `seaborn`, `Plotly`, `Streamlit`  
- **Deployment**: Streamlit dashboard for visualizing results

---

## 🧪 Methodology & Steps

### 1️⃣ Data Preparation
- Load & clean `single_genre_artists.csv`
- Drop unnecessary columns (e.g., track/artist names)
- Normalize numerical features using `StandardScaler`

### 2️⃣ Feature Selection
Used features that describe sound and emotion:
- `danceability`, `energy`, `valence`, `tempo`, `acousticness`, `speechiness`, etc.

Applied PCA (2D) for **visualization**, not for clustering input.

---

### 3️⃣ Clustering Algorithms

- **K-Means** (Elbow & Silhouette methods used to determine optimal `k`)
- **DBSCAN** (Density-based, tuned using K-distance graph)
- **Agglomerative Hierarchical Clustering** (Visualized with dendrograms)

---

### 4️⃣ Evaluation Metrics

| Metric               | Purpose                                      |
|----------------------|---------------------------------------------|
| Silhouette Score     | Measures cluster cohesion & separation       |
| PCA Visuals          | Human-friendly interpretation of clusters   |
| Heatmaps             | Shows average feature values per cluster    |

---

### 5️⃣ Visualization

- 📊 **Cluster Heatmaps**
- 🌀 **PCA 2D Scatter Plots**
- 🎶 **Genre Distribution per Cluster**

---

## 📈 Results

Clusters uncovered distinct track groupings such as:

- 🎉 **Party Tracks**: High danceability & energy  
- 🎻 **Acoustic & Chill**: High acousticness, low tempo  
- 🧠 **Vocal/Speechy**: High speechiness, moderate energy  

**Silhouette Scores (Indicative):**
- **K-Means**: `~0.37+`
- **Hierarchical**: `~0.34+`
- **DBSCAN**: Variable (depends on noise/outlier points)

---

## 📊 Streamlit Dashboard

An interactive dashboard is available in `Amazon music.py`.

| Feature               | Description                                |
|------------------------|--------------------------------------------|
| 🎛️ Sidebar           | Filter by clustering method                 |
| 📊 Metrics            | Real-time stats (song count, genres, clusters) |
| 🌀 PCA Plot           | Interactive 2D scatter of clusters          |
| 🔥 Heatmap            | Visualizes average feature per cluster      |
| 🎶 Genre Distribution | Top genres represented in each cluster      |
| ⬇️ CSV Export        | Download the clustered dataset              |

