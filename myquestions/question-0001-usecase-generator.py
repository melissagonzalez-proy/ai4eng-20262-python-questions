import numpy as np
import pandas as pd
import random

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture

def generar_caso_de_uso_seleccionar_gmm_por_bic():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para un proceso que:
      - Imputa con mediana + escala con StandardScaler
      - Ajusta múltiples GaussianMixture para distintos k
      - Selecciona best_k por BIC mínimo
      - Predice etiquetas con el mejor y re-ajusta un pipeline final
    OUTPUT: (labels, best_k, bic_df, pipe)
    """

    # ---------------------------------------------------------
    # 1. Configuración aleatoria de dimensiones / parámetros
    # ---------------------------------------------------------
    n_rows       = random.randint(250, 600)
    n_features   = random.randint(3, 8)
    k_candidatos = sorted(set(np.random.default_rng().integers(1, 9, size=5).tolist()))
    covariance_type = random.choice(["full", "tied", "diag", "spherical"])
    random_state = random.randint(0, 10**6)

    # ---------------------------------------------------------
    # 2. Generar datos aleatorios (clústeres + NaNs)
    # ---------------------------------------------------------
    rng = np.random.default_rng(random_state)
    k_true = random.randint(2, 5)

    sizes = rng.integers(low=max(20, n_rows // (2*k_true)),
                         high=max(30, n_rows // k_true + 20),
                         size=k_true)
    while sizes.sum() < n_rows:
        sizes[rng.integers(0, k_true)] += 1
    while sizes.sum() > n_rows:
        idx = rng.integers(0, k_true)
        if sizes[idx] > 1:
            sizes[idx] -= 1

    centers = rng.normal(0, 4, size=(k_true, n_features))
    bloques = []
    for i, sz in enumerate(sizes):
        A = rng.normal(0, 0.7, size=(n_features, n_features))
        cov = A @ A.T + np.eye(n_features) * rng.uniform(0.2, 1.0)
        part = rng.multivariate_normal(centers[i], cov, size=sz)
        bloques.append(part)
    X_full = np.vstack(bloques)
    rng.shuffle(X_full)

    # NaNs ~5%
    nan_mask = rng.random(X_full.shape) < 0.05
    X_full[nan_mask] = np.nan

    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X_full, columns=cols)

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),           # copia defensiva
        "features": None,          # usar todas las numéricas
        "k_values": tuple(k_candidatos),
        "covariance_type": covariance_type,
        "random_state": random_state
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado
    # ---------------------------------------------------------
    # A. Selección de columnas y preprocesamiento
    feats = input_data["df"].select_dtypes(include=[np.number]).columns.tolist()
    X = input_data["df"][feats].to_numpy()

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_sca = scaler.fit_transform(X_imp)

    # B. Ajustar GMM para cada k y calcular BIC
    bic_rows = []
    gmms = {}
    for k in input_data["k_values"]:
        gmm = GaussianMixture(
            n_components=int(k),
            covariance_type=input_data["covariance_type"],
            random_state=int(input_data["random_state"]),
            n_init=2
        )
        gmm.fit(X_sca)
        bic = gmm.bic(X_sca)
        gmms[int(k)] = gmm
        bic_rows.append((int(k), float(bic)))

    bic_df = pd.DataFrame(bic_rows, columns=["k", "bic"]).sort_values("k").reset_index(drop=True)
    best_k = int(bic_df.sort_values("bic", ascending=True).iloc[0]["k"])
    best_gmm = gmms[best_k]

    # C. Etiquetas con el mejor GMM
    labels = best_gmm.predict(X_sca).astype(int)

    # D. “Pipeline” final re-ajustado (imputer + scaler + GMM best_k)
    final_gmm = GaussianMixture(
        n_components=best_k,
        covariance_type=input_data["covariance_type"],
        random_state=int(input_data["random_state"]),
        n_init=2
    )
    pipe = Pipeline(steps=[
        ("imputer", imputer),
        ("scaler", scaler),
        ("gmm", final_gmm)
    ])
    pipe.fit(X)  # re-ajuste completo sobre los datos originales

    output_data = (labels, best_k, bic_df, pipe)
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_seleccionar_gmm_por_bic()
    labels, best_k, bic_df, _ = salida_esperada
    print("\n=== [GMM-BIC] INPUT ===")
    print({k: (f"DataFrame{entrada['df'].shape}" if k == "df" else v) for k, v in entrada.items()})
    print("=== [GMM-BIC] OUTPUT ESPERADO ===")
    print("best_k:", best_k, "| labels.shape:", labels.shape)
    print(bic_df.head())