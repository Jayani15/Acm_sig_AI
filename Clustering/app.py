import streamlit as st
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

st.set_page_config(layout="wide", page_title="Clustering Land Mines")

@st.cache_data
def load_data():
    land_mines = fetch_ucirepo(id=763)
    X = land_mines.data.features
    y = land_mines.data.targets
    df = pd.concat([X, y], axis=1)
    df.columns = ["voltage", "height", "soiltype", "minetype"]
    return df

def gen_augmented_data(df, n_per_cluster=50, noise=0.05, cluster_col="cluster"):
    centers = df.groupby(cluster_col)[["voltage","height","soiltype"]].mean()
    rows = []
    for c, vals in centers.iterrows():
        for _ in range(n_per_cluster):
            row = vals + np.random.normal(0, noise, size=3) * (vals + 1)
            rows.append([row[0], row[1], row[2], df['minetype'].mode()[0], c])
    aug = pd.DataFrame(rows, columns=["voltage","height","soiltype","minetype",cluster_col])
    return aug

df = load_data()
st.title("Minimal Streamlit: Clustering - Land Mines")
st.markdown("Use the controls to run KMeans, view elbow, silhouette and optional augmentation.")

with st.sidebar:
    n_clusters = st.slider("KMeans clusters", 2, 8, 4)
    run_elbow = st.checkbox("Show elbow (1..10)", True)
    augment = st.checkbox("Generate augmented data", False)
    n_aug = st.number_input("Augmented points per cluster", min_value=10, max_value=1000, value=100, step=10)
    noise = st.slider("Augmentation noise level", 0.01, 1.0, 0.1)

st.subheader("Raw data sample")
st.dataframe(df.head())

features = df[["voltage","height","soiltype"]]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

if run_elbow:
    inertias = []
    ks = list(range(1, 11))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(scaled)
        inertias.append(km.inertia_)
    fig_elbow = px.line(x=ks, y=inertias, labels={"x":"k","y":"Inertia"}, title="Elbow")
    st.plotly_chart(fig_elbow, use_container_width=True)

km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(scaled)
df["cluster"] = km.labels_
sil = silhouette_score(scaled, df["cluster"])
st.metric("Silhouette score (original)", f"{sil:.4f}")

fig = px.scatter_3d(df, x="voltage", y="height", z="soiltype",
                    color="cluster", title="KMeans clusters (3D)", width=900, height=600)
st.plotly_chart(fig, use_container_width=True)

if augment:
    aug = gen_augmented_data(df, n_per_cluster=n_aug, noise=noise, cluster_col="cluster")
    aug_scaled = scaler.transform(aug[["voltage","height","soiltype"]])
    km_aug = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(aug_scaled)
    aug["cluster"] = km_aug.labels_
    sil_aug = silhouette_score(aug_scaled, aug["cluster"])
    st.metric("Silhouette score (augmented)", f"{sil_aug:.4f}")
    combined = pd.concat([df.assign(source="original"), aug.assign(source="augmented")], ignore_index=True)
    fig2 = px.scatter_3d(combined, x="voltage", y="height", z="soiltype",
                        color="source", symbol="cluster", title="Original vs Augmented")
    st.plotly_chart(fig2, use_container_width=True)