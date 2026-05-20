import numpy as np
from sklearn.cluster import KMeans

def calcular_inercia_clusters(X: np.ndarray, n_clusters: int) -> float:
    """
    Calcula la inercia (suma de distancias cuadradas a los centros) para K-Means.
    
    Parámetros:
    -----------
    X : np.ndarray
        Matriz de datos de forma (n_muestras, n_características)
    n_clusters : int
        Número de clusters a formar
    
    Retorna:
    --------
    float
        Valor de inercia del modelo K-Means
    """
    # Inicializar y entrenar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Retornar la inercia (suma de distancias cuadradas a los centros más cercanos)
    return kmeans.inertia_