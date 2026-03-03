"""
Misión: detectar_outliers_lof (LOF con imputación, escalado y predicción en test)

Este script incluye:
1) SOLUCIÓN: detectar_outliers_lof
2) GENERADOR DE CASOS: generar_caso_de_uso_detectar_outliers_lof
   - *** AQUÍ se construye el INPUT ***
   - *** AQUÍ se calcula el OUTPUT esperado (llamando a la función solución) ***
3) EJEMPLOS DE USO:
   - Llamada directa con **args
   - Validación básica (formas, valores únicos y rangos de scores)

Requisitos: numpy, pandas, scikit-learn
"""

# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor



# =========================
# 1) GENERADOR DE CASOS
# =========================
def generar_caso_de_uso_detectar_outliers_lof():
    """
    Genera un caso aleatorio para detectar_outliers_lof.

    Retorna
    -------
    args : dict
        Diccionario con los argumentos para llamar detectar_outliers_lof(**args).
        *** AQUÍ se construye el INPUT ***
    output : tuple
        (y_pred, scores, pipe) -> lo que se espera que retorne con esos args.
        *** AQUÍ se calcula el OUTPUT esperado (ejecutando la función solución) ***
    """
    rng = np.random.default_rng()

    # --- Config aleatoria ---
    # nº de filas entre 180 y 500; nº de features entre 3 y 8
    n_rows = int(rng.integers(180, 500))
    n_features = int(rng.integers(3, 9))
    test_size = float(rng.choice([0.2, 0.25, 0.3]))
    random_state = int(rng.integers(0, 10**6))

    # Elegimos n_neighbors en función de un tamaño de train esperado
    train_rows_aprox = int((1 - test_size) * n_rows)
    n_neighbors = int(rng.integers(10, max(11, min(50, train_rows_aprox // 3))))

    contamination = float(rng.uniform(0.03, 0.12))  # 3%..12% esperado

    # --- Datos con densidades distintas ---
    # Creamos varios "grupos" con distintas densidades (clusters)
    k_clusters = int(rng.integers(2, 5))  # entre 2 y 4 zonas
    centers = rng.normal(0, 5, size=(k_clusters, n_features))

    # Tamaños por cluster (suman n_rows)
    sizes = rng.integers(low=max(15, n_rows // (3 * k_clusters)),
                         high=max(16, n_rows // k_clusters + 20),
                         size=k_clusters)
    # Ajustar suma a n_rows
    while sizes.sum() < n_rows:
        sizes[rng.integers(0, k_clusters)] += 1
    if sizes.sum() > n_rows:
        diff = sizes.sum() - n_rows
        for _ in range(diff):
            idx = rng.integers(0, k_clusters)
            if sizes[idx] > 1:
                sizes[idx] -= 1

    chunks = []
    for i, sz in enumerate(sizes):
        # Densidad distinta por cluster: varianza aleatoria
        scale = rng.uniform(0.6, 2.2)
        chunk = centers[i] + rng.normal(0, scale, size=(sz, n_features))
        chunks.append(chunk)

    X_full = np.vstack(chunks)
    rng.shuffle(X_full)

    # Inyectamos algunos NaNs (~5%) para probar imputación
    nan_mask = rng.random(X_full.shape) < 0.05
    X_with_nans = X_full.copy()
    X_with_nans[nan_mask] = np.nan

    # DataFrame final
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X_with_nans, columns=cols)

    # *** AQUÍ se construye el INPUT ***
    args = {
        "df": df,
        "features": None,             # usar todas las numéricas
        "n_neighbors": n_neighbors,
        "contamination": contamination,
        "test_size": test_size,
        "random_state": random_state
    }

    # *** AQUÍ se calcula el OUTPUT esperado ***
    output = detectar_outliers_lof(**args)

    return args, output


# =========================
# 2) EJEMPLOS DE USO
# =========================
if __name__ == "__main__":
    print(f"scikit-learn version: {sklearn.__version__}")

    # 3.1) Generar un caso y ver INPUT/OUTPUT
    args, output_esp = generar_caso_de_uso_detectar_outliers_lof()

    print("\n== INPUT generado (args) ==")
    resumen = {
        k: (f"DataFrame{args['df'].shape}" if k == "df" else v)
        for k, v in args.items()
    }
    print(resumen)

    print("\n== OUTPUT esperado por la solución ==")
    y_pred_exp, scores_exp, pipe_exp = output_esp
    uniques, counts = np.unique(y_pred_exp, return_counts=True)
    print("y_pred valores únicos:", dict(zip(uniques.tolist(), counts.tolist())))
    print("scores: shape", scores_exp.shape, "| rango aproximado:",
          (float(np.nanmin(scores_exp)), float(np.nanmax(scores_exp))))
    print("pipeline:", type(pipe_exp).__name__)
