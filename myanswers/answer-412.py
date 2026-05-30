import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def segmentar_paisajes_sonoros(df, n_clusters):

    # Eliminar timestamp
    X = df.drop(columns=["timestamp"])

    # Imputar valores faltantes
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # Escalar datos
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # PCA a 2 componentes
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    varianza_explicada_pca = [
        float(v) for v in pca.explained_variance_ratio_
    ]

    # Clustering
    modelo_cluster = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )

    labels = modelo_cluster.fit_predict(X_pca)

    # Silhouette
    if len(np.unique(labels)) < 2:
        silhouette = -1.0
    else:
        silhouette = round(
            float(
                silhouette_score(
                    X_scaled,
                    labels,
                    metric="euclidean"
                )
            ),
            4
        )

    return {
        "labels": labels,
        "silhouette": silhouette,
        "varianza_explicada_pca": varianza_explicada_pca
    }
