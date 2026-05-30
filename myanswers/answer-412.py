import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def segmentar_paisajes_sonoros(df, n_clusters):
    
    # Columnas a usar para clustering (excluir timestamp)
    feature_cols = ['intensidad_media', 'intensidad_max', 
                    'frecuencia_media', 'frecuencia_dominante', 
                    'diversidad_espectral']
    
    # Extraer características
    X = df[feature_cols].values
    
    # Manejar valores nulos si existen (reemplazar con mediana)
    if np.any(pd.isnull(X)):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    # Estandarizar características 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, 
                    random_state=42, 
                    n_init=10,
                    max_iter=300)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calcular silhouette score (solo si hay más de 1 cluster y suficientes muestras)
    if n_clusters > 1 and len(np.unique(labels)) > 1 and len(X_scaled) > n_clusters:
        silhouette = silhouette_score(X_scaled, labels)
    else:
        silhouette = -1.0  # No aplicable
    
    # PCA para análisis de varianza explicada
    pca = PCA()
    pca.fit(X_scaled)
    varianza_explicada = pca.explained_variance_ratio_.tolist()
    
    # Resultados
    return {
        'labels': labels,
        'silhouette': silhouette,
        'varianza_explicada_pca': varianza_explicada
    }

