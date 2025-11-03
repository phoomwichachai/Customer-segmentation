# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Data ---
data = pd.read_csv("customer_data.csv")
features = ['Age', 'Annual Income', 'Spending Score']
X = data[features]

# --- 2. Standardize Data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. PCA for Visualization ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- 4. Function to Evaluate Clustering ---
def evaluate_clustering(labels, X_scaled):
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
    else:
        silhouette = ch_score = db_score = np.nan
    return silhouette, ch_score, db_score

# --- 5. KMeans Grid Search ---
best_k = 0
best_score = -1
k_results = []

for k in range(2, 11):  # ลอง cluster 2-10
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette, ch_score, db_score = evaluate_clustering(labels, X_scaled)
    k_results.append((k, silhouette, ch_score, db_score))
    if silhouette > best_score:
        best_score = silhouette
        best_k = k
        best_labels_kmeans = labels

print(f"Best KMeans clusters: {best_k} with Silhouette = {best_score:.4f}")

# --- 6. Agglomerative Grid Search ---
best_k_agg = 0
best_score_agg = -1
agg_results = []

for k in range(2, 11):
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X_scaled)
    silhouette, ch_score, db_score = evaluate_clustering(labels, X_scaled)
    agg_results.append((k, silhouette, ch_score, db_score))
    if silhouette > best_score_agg:
        best_score_agg = silhouette
        best_k_agg = k
        best_labels_agg = labels

print(f"Best Agglomerative clusters: {best_k_agg} with Silhouette = {best_score_agg:.4f}")

# --- 7. DBSCAN ---
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
silhouette_db, ch_db, db_db = evaluate_clustering(labels_dbscan, X_scaled)
print(f"DBSCAN: Silhouette = {silhouette_db:.4f}, CH = {ch_db:.4f}, DB = {db_db:.4f}")

# --- 8. Visualization Function ---
def plot_clusters(X_pca, labels, title):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette="tab10", s=60)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title='Cluster')
    plt.show()

plot_clusters(X_pca, best_labels_kmeans, f"KMeans (k={best_k})")
plot_clusters(X_pca, best_labels_agg, f"Agglomerative (k={best_k_agg})")
plot_clusters(X_pca, labels_dbscan, "DBSCAN")

# --- 9. Summary Table ---
summary = pd.DataFrame({
    'Model': ['KMeans', 'Agglomerative', 'DBSCAN'],
    'Best Clusters': [best_k, best_k_agg, 'Variable'],
    'Silhouette': [best_score, best_score_agg, silhouette_db],
    'Calinski-Harabasz': [k_results[best_k-2][2], agg_results[best_k_agg-2][2], ch_db],
    'Davies-Bouldin': [k_results[best_k-2][3], agg_results[best_k_agg-2][3], db_db]
})
print(summary)
