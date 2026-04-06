"""
Misión: detectar_outliers_lof (LOF con imputación, escalado y predicción en test)

Este script incluye:
1) SOLUCIÓN: detectar_outliers_lof
2) GENERADOR DE CASOS (alineado a tu plantilla): generar_caso_de_uso_detectar_outliers_lof__manual
   - *** AQUÍ se construye el INPUT ***
   - *** AQUÍ se calcula el OUTPUT esperado (replicando la lógica MANUALMENTE, sin llamar a la solución) ***
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
# 1) FUNCIÓN SOLUCIÓN (sin cambios)
# =========================
def detectar_outliers_lof(
    df,
    features=None,
    n_neighbors=20,
    contamination=0.05,
    test_size=0.2,
    random_state=42
):
    """
    Detecta outliers con LocalOutlierFactor (LOF) tras imputar y escalar.
    Usa novelty=True para entrenar en train y predecir en test.

    Devuelve
    --------
    y_pred : numpy.ndarray
        Etiquetas en test ({1 = normal, -1 = outlier}).
    scores : numpy.ndarray
        decision_function en test (valores mayores => más normal).
    pipe : sklearn.pipeline.Pipeline
        Pipeline ajustado: imputación + escalado + LOF (novelty=True).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas.DataFrame")

    # 1) Selección de columnas
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(features) == 0:
        raise ValueError("No hay columnas numéricas disponibles para detectar outliers.")
    X = df[features].copy()

    # 2) División train/test
    X_train, X_test = train_test_split(
        X, test_size=float(test_size), random_state=int(random_state)
    )

    # Ajustar n_neighbors a tamaño de train
    max_valid = max(5, min(int(n_neighbors), len(X_train) - 1))
    if max_valid < 2:
        raise ValueError("Muy pocas muestras en entrenamiento para aplicar LOF.")
    n_neighbors_adj = max_valid

    # 3) Pipeline (imputación + escalado + LOF con novelty=True)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors_adj,
        contamination=float(contamination),
        novelty=True
    )
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lof", lof)
    ])

    # 4) Entrenamiento y predicción
    pipe.fit(X_train)
    y_pred = pipe.predict(X_test).astype(int)                 # {1, -1}
    scores = pipe.decision_function(X_test).astype(float)     # mayor => más “normal”

    return y_pred, scores, pipe


# =========================
# 2) GENERADOR DE CASOS (alineado a tu plantilla)
# =========================
def generar_caso_de_uso_detectar_outliers_lof__manual():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la misión de LOF, **replicando manualmente** la lógica esperada
    (sin llamar a la función solución).

    Retorna
    -------
    input_data : dict
        *** AQUÍ se construye el INPUT ***
    output_data : tuple
        (y_pred, scores, pipe)  -> *** AQUÍ se calcula el OUTPUT esperado ***
    """
    # ---------------------------------------------------------
    # 1. Configuración aleatoria de dimensiones / parámetros
    # ---------------------------------------------------------
    rng = np.random.default_rng()
    n_rows = int(rng.integers(180, 500))   # nº de filas entre 180 y 500
    n_features = int(rng.integers(3, 9))   # nº de features entre 3 y 8
    test_size = float(rng.choice([0.2, 0.25, 0.3]))
    random_state = int(rng.integers(0, 10**6))

    # Elegimos n_neighbors en función de un tamaño de train esperado
    train_rows_aprox = int((1 - test_size) * n_rows)
    n_neighbors = int(rng.integers(10, max(11, min(50, max(10, train_rows_aprox // 3)))))

    contamination = float(rng.uniform(0.03, 0.12))  # 3%..12% esperado

    # ---------------------------------------------------------
    # 2. Generación de datos con densidades diferentes (clusters) + NaNs
    # ---------------------------------------------------------
    k_clusters = int(rng.integers(2, 5))  # entre 2 y 4 zonas
    centers = rng.normal(0, 5, size=(k_clusters, n_features))

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
        scale = rng.uniform(0.6, 2.2)  # varianza por cluster
        chunk = centers[i] + rng.normal(0, scale, size=(sz, n_features))
        chunks.append(chunk)

    X_full = np.vstack(chunks)
    rng.shuffle(X_full)

    # Inyectar algunos NaNs (~5%) para probar imputación
    nan_mask = rng.random(X_full.shape) < 0.05
    X_with_nans = X_full.copy()
    X_with_nans[nan_mask] = np.nan

    # DataFrame final
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X_with_nans, columns=cols)

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "features": None,             # usar todas las numéricas
        "n_neighbors": n_neighbors,
        "contamination": contamination,
        "test_size": test_size,
        "random_state": random_state
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (replicando la lógica MANUALMENTE)
    # ---------------------------------------------------------
    # A) Selección de columnas y split
    feats = input_data["df"].select_dtypes(include=[np.number]).columns.tolist()
    if len(feats) == 0:
        raise ValueError("No hay columnas numéricas disponibles para detectar outliers.")
    X = input_data["df"][feats].to_numpy()

    X_train, X_test = train_test_split(
        X, test_size=float(input_data["test_size"]), random_state=int(input_data["random_state"])
    )

    # Ajuste de n_neighbors a tamaño de train
    max_valid = max(5, min(int(input_data["n_neighbors"]), len(X_train) - 1))
    if max_valid < 2:
        raise ValueError("Muy pocas muestras en entrenamiento para aplicar LOF.")
    n_neighbors_adj = max_valid

    # B) Preprocesamiento (imputación + escalado) y LOF
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors_adj,
        contamination=float(input_data["contamination"]),
        novelty=True
    )

    # Ajuste manual del “pipeline”
    X_train_imp = imputer.fit_transform(X_train)
    X_train_sca = scaler.fit_transform(X_train_imp)
    lof.fit(X_train_sca)

    # Inferencia en test
    X_test_imp = imputer.transform(X_test)
    X_test_sca = scaler.transform(X_test_imp)
    y_pred = lof.predict(X_test_sca).astype(int)               # {1, -1}
    scores = lof.decision_function(X_test_sca).astype(float)   # mayor => más normal

    # C) Construcción del pipe “entrenado”
    pipe = Pipeline(steps=[
        ("imputer", imputer),
        ("scaler", scaler),
        ("lof", lof)
    ])

    output_data = (y_pred, scores, pipe)
    return input_data, output_data


# =========================
# 3) EJEMPLOS DE USO
# =========================
if __name__ == "__main__":
    print(f"scikit-learn version: {sklearn.__version__}")

    # 3.1) Generar un caso y ver INPUT/OUTPUT esperado (manual)
    args, output_esp = generar_caso_de_uso_detectar_outliers_lof__manual()

    print("\n== INPUT generado (args) ==")
    resumen = {
        k: (f"DataFrame{args['df'].shape}" if k == "df" else v)
        for k, v in args.items()
    }
    print(resumen)

    print("\n== OUTPUT esperado (calculado manualmente) ==")
    y_pred_exp, scores_exp, pipe_exp = output_esp
    uniques, counts = np.unique(y_pred_exp, return_counts=True)
    print("y_pred valores únicos:", dict(zip(uniques.tolist(), counts.tolist())))
    print("scores: shape", scores_exp.shape, "| rango aproximado:",
          (float(np.nanmin(scores_exp)), float(np.nanmax(scores_exp))))
    print("pipeline:", type(pipe_exp).__name__)

    # 3.2) (Opcional) Ejecutar la función solución con **args y comparar tamaños
    y_pred_run, scores_run, pipe_run = detectar_outliers_lof(**args)

    print("\n== OUTPUT al ejecutar detectar_outliers_lof(**args) ==")
    uniques2, counts2 = np.unique(y_pred_run, return_counts=True)
    print("y_pred valores únicos:", dict(zip(uniques2.tolist(), counts2.tolist())))
    print("scores: shape", scores_run.shape)
    print("pipeline:", type(pipe_run).__name__)

    # 3.3) Validación básica
    assert y_pred_run.shape == y_pred_exp.shape, "Las formas de y_pred no coinciden"
    assert scores_run.shape == scores_exp.shape, "Las formas de scores no coinciden"
    assert set(np.unique(y_pred_run)).issubset({-1, 1}), "y_pred debe contener solo -1 y 1"
    print("\n✔ Validación básica superada: formas consistentes y valores válidos.")

    # 3.4) Demostración de aleatoriedad (2 casos adicionales)
    print("\n== Generando 2 casos aleatorios adicionales ==")
    for i in range(2):
        a, o = generar_caso_de_uso_detectar_outliers_lof__manual()
        yp, sc, _ = o
        fr_out = float(np.mean(yp == -1))
        print(f" Caso {i+1}: df{a['df'].shape}, k={a['n_neighbors']}, "
              f"contamination={a['contamination']:.3f}, outliers_detectados={fr_out:.3f}")
