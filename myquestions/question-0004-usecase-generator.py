import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


def generar_caso_de_uso_detectar_outliers_lof():
    """
    Genera un caso de uso para detección de outliers con LOF.

    Retorna:
    --------
    input_data : dict
        Diccionario con los argumentos de entrada.

    output_data : tuple
        (
            y_pred,
            scores,
            pipe
        )
    """

    # =====================================================
    # 1. Configuración aleatoria
    # =====================================================
    rng = np.random.default_rng()

    n_rows = int(rng.integers(180, 500))
    n_features = int(rng.integers(3, 9))

    test_size = float(rng.choice([0.2, 0.25, 0.3]))
    random_state = int(rng.integers(0, 1_000_000))

    train_rows_aprox = int((1 - test_size) * n_rows)

    n_neighbors = int(
        rng.integers(
            10,
            max(11, min(50, max(10, train_rows_aprox // 3)))
        )
    )

    contamination = float(rng.uniform(0.03, 0.12))

    # =====================================================
    # 2. Generación de datos sintéticos
    # =====================================================
    k_clusters = int(rng.integers(2, 5))

    centers = rng.normal(
        0,
        5,
        size=(k_clusters, n_features)
    )

    sizes = rng.integers(
        low=max(15, n_rows // (3 * k_clusters)),
        high=max(16, n_rows // k_clusters + 20),
        size=k_clusters
    )

    # Ajustar tamaños para que sumen n_rows
    while sizes.sum() < n_rows:
        sizes[rng.integers(0, k_clusters)] += 1

    while sizes.sum() > n_rows:
        idx = rng.integers(0, k_clusters)
        if sizes[idx] > 1:
            sizes[idx] -= 1

    chunks = []

    for i, sz in enumerate(sizes):
        scale = rng.uniform(0.6, 2.2)

        chunk = centers[i] + rng.normal(
            0,
            scale,
            size=(sz, n_features)
        )

        chunks.append(chunk)

    X_full = np.vstack(chunks)

    rng.shuffle(X_full)

    # =====================================================
    # 3. Introducción de NaNs
    # =====================================================
    nan_mask = rng.random(X_full.shape) < 0.05

    X_with_nans = X_full.copy()

    X_with_nans[nan_mask] = np.nan

    # =====================================================
    # 4. Construcción del DataFrame
    # =====================================================
    cols = [f"f{i}" for i in range(n_features)]

    df = pd.DataFrame(
        X_with_nans,
        columns=cols
    )

    # =====================================================
    # 5. Construcción del INPUT
    # =====================================================
    input_data = {
        "df": df.copy(),
        "features": None,
        "n_neighbors": n_neighbors,
        "contamination": contamination,
        "test_size": test_size,
        "random_state": random_state
    }

    # =====================================================
    # 6. Lógica manual esperada
    # =====================================================

    # Selección de columnas numéricas
    features = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(features) == 0:
        raise ValueError(
            "No hay columnas numéricas disponibles."
        )

    X = df[features]

    # División train/test
    X_train, X_test = train_test_split(
        X,
        test_size=test_size,
        random_state=random_state
    )

    # Ajuste defensivo de n_neighbors
    n_neighbors_adj = min(
        n_neighbors,
        len(X_train) - 1
    )

    if n_neighbors_adj < 2:
        raise ValueError(
            "Muy pocas muestras para aplicar LOF."
        )

    # =====================================================
    # 7. Construcción del pipeline
    # =====================================================
    pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "lof",
            LocalOutlierFactor(
                n_neighbors=n_neighbors_adj,
                contamination=contamination,
                novelty=True
            )
        )
    ])

    # =====================================================
    # 8. Entrenamiento
    # =====================================================
    pipe.fit(X_train)

    # =====================================================
    # 9. Predicción
    # =====================================================
    y_pred = pipe.predict(X_test).astype(int)

    scores = pipe.decision_function(X_test).astype(float)

    # =====================================================
    # 10. OUTPUT esperado
    # =====================================================
    output_data = (
        y_pred,
        scores,
        pipe
    )

    return input_data, output_data
