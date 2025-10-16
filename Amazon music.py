# ==========================================
# 💎 Amazon Music Clustering 
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px

# ------------------------------------------------------------
# 1️⃣ PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Music Cluster Visualizer (Pro)",
    layout="wide",
    page_icon="🎧"
)

# ------------------------------------------------------------
# 2️⃣ PREMIUM LIGHT THEME STYLING
# ------------------------------------------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #fefefe;
    color: #212529;
    font-family: 'Poppins', sans-serif;
}

h1, h2, h3, h4, h5 { color: #222; font-weight: 600; }
h1 { background: linear-gradient(90deg, #007bff, #00bcd4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #f0f9ff, #e3f2fd); color: #212529; border-right: 1px solid #dee2e6; }

.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #007bff, #00bcd4);
    color: white; border-radius: 10px; font-weight: 600; border: none; padding: 0.6em 1em;
    transition: all 0.2s ease-in-out; box-shadow: 0 2px 5px rgba(0,0,0,0.15);
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(90deg, #0056b3, #0097a7); transform: scale(1.03); box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}

div[data-baseweb="select"] > div {
    background-color: #f1f3f5 !important; color: #212529 !important; border-radius: 8px; border: 1px solid #ced4da;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
}

.stMetric { background: linear-gradient(135deg, #f9f9f9, #e8f5fe); border-radius: 12px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.stDataFrame { background-color: #ffffff !important; border-radius: 10px !important; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }

.block-container { padding-top: 1rem; padding-bottom: 2rem; }
hr { border: 1px solid #e3f2fd; }
footer, .stCaption, .stMarkdown small { color: #6c757d !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 3️⃣ HEADER
# ------------------------------------------------------------
st.markdown(
    "<h1 style='text-align: center; background: linear-gradient(90deg, #007bff, #00bcd4); "
    "-webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;'>"
    "🎧 Amazon Music Clustering Dashboard"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<div style='text-align: center; color: #212529; font-weight: 600;'>"
    "Analyze & visualize audio and artist features using advanced clustering algorithms."
    "</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------------------------------------------------
# 4️⃣ LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_music_clusters_all_methods.csv")
    return df

df = load_data()

# Sidebar filter
st.sidebar.header("⚙️ Filter Options")
method = st.sidebar.selectbox("Select Clustering Method", ["K-Means", "DBSCAN", "Hierarchical"])
cluster_col = {
    "K-Means": "cluster",
    "DBSCAN": "cluster_dbscan",
    "Hierarchical": "cluster_hc"
}[method]

# Drop NaN clusters (for HC/DBSCAN)
df_vis = df.dropna(subset=[cluster_col]).copy()
df_vis[cluster_col] = df_vis[cluster_col].astype(int)

# ------------------------------------------------------------
# 5️⃣ DATA OVERVIEW
# ------------------------------------------------------------
st.subheader("📊 Dataset Overview")
st.dataframe(df_vis.head(10))

col1, col2, col3 = st.columns(3)
col1.metric("🎵 Number of Songs", len(df_vis))
col2.metric("🎯 Number of Clusters", df_vis[cluster_col].nunique())
col3.metric("🎹 Genres", df_vis['genres'].nunique())

# ------------------------------------------------------------
# 6️⃣ CLUSTER DISTRIBUTION
# ------------------------------------------------------------
st.subheader(f"🎨 {method} Cluster Distribution")
cluster_counts = df_vis[cluster_col].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(
    x=cluster_counts.index,
    y=cluster_counts.values,
    hue=cluster_counts.index,
    ax=ax,
    palette="crest",
    dodge=False,
    legend=False
)
ax.set_xlabel("Cluster ID", color='#212529')
ax.set_ylabel("Number of Songs", color='#212529')
ax.set_title(f"{method} – Cluster Size Distribution", color='#212529')
ax.tick_params(colors='#212529')
sns.despine()
st.pyplot(fig)

# ------------------------------------------------------------
# 7️⃣ FEATURE COMPARISON PER CLUSTER
# ------------------------------------------------------------
st.subheader("🎼 Average Feature Values per Cluster")
features = ['danceability', 'energy', 'valence', 'tempo']
cluster_profile = df_vis.groupby(cluster_col)[features].mean()

fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(cluster_profile, annot=True, cmap='PuBuGn', fmt=".2f", ax=ax)
ax.set_title(f"{method} – Feature Profile Heatmap", color='#212529')
st.pyplot(fig)

# ------------------------------------------------------------
# 8️⃣ PCA VISUALIZATION (2D)
# ------------------------------------------------------------
st.subheader("🌀 PCA Visualization (2D Projection)")
numeric_cols = ['danceability', 'energy', 'valence', 'tempo']
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_vis[numeric_cols])
df_vis['pca1'], df_vis['pca2'] = pca_result[:,0], pca_result[:,1]

fig_pca = px.scatter(
    df_vis,
    x='pca1',
    y='pca2',
    color=df_vis[cluster_col].astype(str),
    hover_data=['genres'],
    title=f"{method} – PCA Cluster Visualization",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_pca.update_layout(paper_bgcolor='#ffffff', plot_bgcolor='#fafafa', font=dict(color="#212529"))

# Use `config` instead of deprecated width
st.plotly_chart(fig_pca, config={"displayModeBar": True, "responsive": True})

# ------------------------------------------------------------
# 9️⃣ GENRE CLUSTER DISTRIBUTION
# ------------------------------------------------------------
st.subheader("🎶 Top Genres by Cluster")
top_genres = (
    df_vis.groupby(['genres', cluster_col])
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
    .head(15)
)

fig_genres = px.bar(
    top_genres,
    x='genres',
    y='count',
    color=cluster_col,
    title=f"Top Genres across {method} Clusters",
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig_genres.update_layout(paper_bgcolor='#ffffff', plot_bgcolor='#fafafa', font=dict(color="#212529"))
st.plotly_chart(fig_genres, config={"displayModeBar": True, "responsive": True})

# ------------------------------------------------------------
# 🔟 DOWNLOAD FILTERED RESULTS
# ------------------------------------------------------------
st.subheader("⬇️ Download Filtered Cluster Data")
csv = df_vis.to_csv(index=False).encode('utf-8')
st.download_button(
    "💾 Download CSV",
    csv,
    f"music_clusters_{method.lower()}.csv",
    "text/csv",
    key='download-csv'
)

st.markdown("---")
st.caption("💙 Built with Streamlit Pro | Dataset: Amazon Music Single-Genre Artists")


