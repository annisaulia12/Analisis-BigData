import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation K-Means", layout="wide")
st.title("📊 Customer Segmentation using K-Means")

# ============================================================
# Upload Dataset
# ============================================================

st.sidebar.header("Upload Dataset CSV")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

# Jika user tidak upload file → ambil dataset dari GitHub
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # BACA FILE DEFAULT DARI REPO GITHUB
    df = pd.read_csv("shopping_trends_updated.csv")
    st.sidebar.info("Using default dataset from repository.")
    st.success("Dataset default berhasil dimuat!")
    
# ============================================================
# 1. PEMAHAMAN DATASET
# ============================================================

st.header(" 1. Pemahaman Dataset")

st.write("### Jumlah baris & kolom")
st.write(df.shape)

st.write("### Tipe Data")
st.dataframe(df.dtypes)

st.write("### Statistik Deskriptif")
st.dataframe(df.describe(include="all").T)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
st.write("### Fitur numerik terdeteksi:")
st.success(numeric_cols)

# ============================================================
# 2. EDA
# ============================================================

st.header(" 2. Exploratory Data Analysis (EDA)")

st.write("### Missing Values per Kolom")
st.dataframe(df.isnull().sum())

# ---- Histogram & Boxplot ----
st.subheader("Distribusi Fitur Numerik (Histogram & Boxplot)")

for col in numeric_cols:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df[col], bins=25, ax=ax[0], kde=True)
    ax[0].set_title(f"Histogram - {col}")

    sns.boxplot(x=df[col], ax=ax[1])
    ax[1].set_title(f"Boxplot - {col}")

    st.pyplot(fig)
    plt.clf()

# ---- Correlation Heatmap ----
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
plt.clf()

# ---- Variance ----
st.write("### Variansi Fitur (Heuristik fitur paling berpengaruh):")
st.dataframe(df[numeric_cols].var().sort_values(ascending=False))

# ============================================================
# ANALISIS SIAPA YANG LEBIH SERING BERBELANJA (GENDER)
# ============================================================

st.header(" Analisis Pelanggan Berdasarkan Gender")

if "Gender" in df.columns:

    st.subheader("Jumlah Customer per Gender")
    st.dataframe(df["Gender"].value_counts())

    st.subheader("Rata-rata Nilai Belanja per Gender")
    if "Purchase Amount (USD)" in df.columns:
        st.dataframe(df.groupby("Gender")["Purchase Amount (USD)"].mean())

    st.subheader("Total Belanja per Gender")
    if "Purchase Amount (USD)" in df.columns:
        total_gender = df.groupby("Gender")["Purchase Amount (USD)"].sum().reset_index()

        import plotly.express as px
        fig_gender = px.bar(
            total_gender,
            x="Gender",
            y="Purchase Amount (USD)",
            title="Total Nilai Pembelian Berdasarkan Gender",
            text="Purchase Amount (USD)"
        )
        st.plotly_chart(fig_gender, use_container_width=True)

else:
    st.warning("Dataset tidak memiliki kolom Gender.")


# ============================================================
# 3. PEMILIHAN & PREPROCESSING FITUR UNTUK CLUSTERING
# ============================================================

st.header(" 3. Preprocessing & Pemilihan Fitur Clustering")

# Fitur yang dipakai
selected_features = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases", "Gender"]
st.write("### Fitur yang digunakan untuk clustering:")
st.success(selected_features)

# Pastikan fitur ada
missing = [col for col in selected_features if col not in df.columns]
if missing:
    st.error(f"Kolom berikut tidak ada dalam dataset: {missing}")
    st.stop()

# Backup gender asli
df["Gender_Original"] = df["Gender"]

# Ambil fitur
df_cluster = df[selected_features].copy()

# Isi missing value numerik
df_cluster.fillna(df_cluster.median(numeric_only=True), inplace=True)

# Encoding gender → 0 = Female, 1 = Male
df_cluster = pd.get_dummies(df_cluster, columns=["Gender"], drop_first=True)

# Tambahkan label human-readable
df_cluster["Gender_Label"] = df_cluster["Gender_Male"].apply(lambda x: "Male" if x==1 else "Female")

st.write("### Data setelah encoding Gender (0=Female, 1=Male):")
st.dataframe(df_cluster.head())

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases", "Gender_Male"]])

st.info("Keterangan Gender Encoding:\n0 = Female\n1 = Male")
st.success("Preprocessing selesai!")

# ============================================================
# 4. MENENTUKAN K OPTIMAL
# ============================================================

st.header(" 4. Menentukan Jumlah Cluster (K Optimal)")

K_range = range(2, 11)
inertia_list = []
sil_list = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia_list.append(km.inertia_)
    sil_list.append(silhouette_score(X_scaled, labels))

# Visual Elbow
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(K_range, inertia_list, marker='o')
ax.set_title("Elbow Method")
ax.set_xlabel("K")
ax.set_ylabel("Inertia")
st.pyplot(fig)

# ============================================================
# 5. K-MEANS FINAL
# ============================================================

st.header(" 5. K-Means Final")

k_final = st.slider("Pilih jumlah cluster (K):", 2, 10, 4)

# Silhouette score untuk K 
silhouette_kfinal = silhouette_score(
    X_scaled,
    KMeans(n_clusters=k_final, random_state=42).fit_predict(X_scaled)
)
st.info(f" Silhouette Score untuk K = {k_final} adalah **{silhouette_kfinal:.4f}**")

model = KMeans(n_clusters=k_final, random_state=42)
df["Cluster"] = model.fit_predict(X_scaled)

st.success("Clustering selesai!")
st.write(df["Cluster"].value_counts())

# ============================================================
# 6. VISUALISASI PCA (2D)
# ============================================================

st.header(" 6. Visualisasi Klaster (PCA 2D)")

pca = PCA(n_components=2)
pca_res = pca.fit_transform(X_scaled)

df["PCA1"] = pca_res[:,0]
df["PCA2"] = pca_res[:,1]

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(
    x="PCA1", y="PCA2",
    hue=df["Cluster"].astype(str),
    palette="tab10",
    data=df,
    ax=ax
)
plt.title("Visualisasi Klaster (PCA 2D)")
st.pyplot(fig)

# ============================================================
# 7. RINGKASAN CLUSTER
# ============================================================

st.header(" 7. Karakteristik Rata-rata Tiap Cluster")

summary = df.groupby("Cluster")[["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]].mean()
st.dataframe(summary)

# ============================================================
# 8. DISTRIBUSI GENDER PER CLUSTER
# ============================================================

st.header(" 8. Distribusi Gender per Cluster")

df["Gender_Label"] = df_cluster["Gender_Label"]

gender_cluster = df.groupby(["Cluster", "Gender_Label"]).size().reset_index(name="Count")

# Hilangkan bar kosong
gender_cluster = gender_cluster[gender_cluster["Count"] > 0]

fig3, ax3 = plt.subplots(figsize=(7,4))
sns.barplot(
    data=gender_cluster,
    x="Cluster",
    y="Count",
    hue="Gender_Label",
    dodge=False,
    ax=ax3
)
plt.title("Distribusi Gender per Cluster (Tanpa Jarak Bar)")
st.pyplot(fig3)