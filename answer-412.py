import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def segmentar_paisajes_sonoros(df, n_clusters):
    """
    Segmenta paisajes sonoros mediante clustering y evalúa la calidad.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con columnas: intensidad_media, intensidad_max, 
        frecuencia_media, frecuencia_dominante, diversidad_espectral, timestamp
    n_clusters : int
        Número de clusters a formar
    
    Retorna:
    --------
    dict
        Diccionario con:
        - 'labels': array con asignaciones de cluster
        - 'silhouette': puntaje de silhouette
        - 'varianza_explicada_pca': lista con varianza explicada por componentes PCA
    """
    
    # 1. Seleccionar columnas numéricas (excluyendo timestamp si existe)
    columnas_acusticas = ['intensidad_media', 'intensidad_max', 
                          'frecuencia_media', 'frecuencia_dominante', 
                          'diversidad_espectral']
    
    # Verificar que todas las columnas necesarias existen
    columnas_disponibles = [col for col in columnas_acusticas if col in df.columns]
    if len(columnas_disponibles) < len(columnas_acusticas):
        missing = set(columnas_acusticas) - set(columnas_disponibles)
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    
    X = df[columnas_acusticas].copy()
    
    # 2. Imputar valores faltantes con la media de cada columna
    imputer = SimpleImputer(strategy='mean')
    X_imputado = imputer.fit_transform(X)
    
    # 3. Escalar los datos
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_imputado)
    
    # 4. Aplicar K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_escalado)
    
    # 5. Calcular silhouette score
    if n_clusters >= 2 and len(np.unique(labels)) >= 2:
        silhouette = silhouette_score(X_escalado, labels)
    else:
        silhouette = -1.0
    
    # 6. Calcular varianza explicada por PCA (primeros componentes hasta 95%)
    pca = PCA()
    pca.fit(X_escalado)
    
    # Obtener varianza explicada acumulada
    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
    
    # Encontrar cuántos componentes se necesitan para 95% de varianza
    # Devolver lista de varianza explicada por cada componente
    varianza_explicada_pca = pca.explained_variance_ratio_.tolist()
    
    return {
        'labels': labels,
        'silhouette': silhouette,
        'varianza_explicada_pca': varianza_explicada_pca
    }