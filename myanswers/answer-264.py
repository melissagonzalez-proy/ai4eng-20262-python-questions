import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import contingency_matrix

def agrupar_usuarios(df, n_clusters):
    """
    Agrupa usuarios usando clustering jerárquico aglomerativo.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con columnas: tiempo_uso_diario, numero_interacciones, 
        contenido_consumido, nivel_actividad, tipo_usuario
    n_clusters : int
        Número de clusters a formar
    
    Retorna:
    --------
    tuple
        (df_con_clusters, silhouette_score, pureza)
    """
    
    # 1. Separar la columna tipo_usuario
    if 'tipo_usuario' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'tipo_usuario'")
    
    tipo_usuario = df['tipo_usuario'].copy()
    df_sin_tipo = df.drop('tipo_usuario', axis=1)
    
    # 2. Seleccionar únicamente columnas numéricas
    columnas_numericas = df_sin_tipo.select_dtypes(include=[np.number]).columns
    X = df_sin_tipo[columnas_numericas]
    
    # 3. Imputar valores faltantes con SimpleImputer (estrategia=constant, fill_value=0)
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputado = imputer.fit_transform(X)
    
    # 4. Escalar los datos con StandardScaler
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_imputado)
    
    # 5. Aplicar AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(X_escalado)
    
    # 6. Añadir columna cluster al DataFrame original
    df_con_clusters = df.copy()
    df_con_clusters['cluster'] = clusters
    
    # 7. Calcular Silhouette Score
    # Nota: silhouette_score requiere al menos 2 clusters y n_samples > n_clusters
    if n_clusters >= 2 and len(np.unique(clusters)) >= 2:
        silhouette = silhouette_score(X_escalado, clusters)
    else:
        silhouette = -1  # Valor por defecto cuando no se puede calcular
    
    # 8. Calcular la pureza del clustering
    pureza = _calcular_pureza(tipo_usuario, clusters)
    
    # 9. Devolver tupla
    return (df_con_clusters, silhouette, pureza)


def _calcular_pureza(y_true, y_pred):
    """
    Calcula la pureza del clustering comparando con etiquetas verdaderas.
    
    Pureza = (1/N) * sum_{k} max_j |cluster_k ∩ clase_j|
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas (tipo_usuario)
    y_pred : array-like
        Etiquetas predichas (clusters)
    
    Retorna:
    --------
    float
        Pureza del clustering (entre 0 y 1)
    """
    # Crear matriz de contingencia
    matriz = contingency_matrix(y_true, y_pred)
    
    # Para cada cluster, encontrar la clase mayoritaria
    max_por_cluster = np.max(matriz, axis=0)
    
    # Sumar los máximos y dividir por el total de muestras
    pureza = np.sum(max_por_cluster) / np.sum(matriz)
    
    return pureza