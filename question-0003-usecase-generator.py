import numpy as np
import pandas as pd
import random

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_propagar_etiquetas_label_spreading():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para un proceso que:
      - Usa solo filas con etiqueta conocida para split
      - En train, oculta fracción de etiquetas como -1
      - Imputa mediana + StandardScaler
      - Ajusta LabelSpreading(kernel, gamma)
      - Predice en test y calcula accuracy
    OUTPUT: (y_test, y_pred, accuracy, pipe)
    """

    # ---------------------------------------------------------
    # 1. Configuración aleatoria
    # ---------------------------------------------------------
    n_rows       = random.randint(300, 700)
    n_features   = random.randint(3, 7)
    n_classes    = random.randint(2, 4)
    test_size    = random.choice([0.2, 0.25, 0.3])
    gamma        = random.choice([10.0, 20.0, 30.0])
    unl_frac     = float(np.clip(np.random.uniform(0.4, 0.7), 0.0, 0.9))
    random_state = random.randint(0, 10**6)

    # ---------------------------------------------------------
    # 2. Generar datos (clases, NaNs en X, etiquetas faltantes)
    # ---------------------------------------------------------
    rng = np.random.default_rng(random_state)
    centers = rng.normal(0, 4, size=(n_classes, n_features))

    sizes = rng.integers(low=max(30, n_rows // (2*n_classes)),
                         high=max(35, n_rows // n_classes + 20),
                         size=n_classes)
    while sizes.sum() < n_rows:
        sizes[rng.integers(0, n_classes)] += 1
    while sizes.sum() > n_rows:
        idx = rng.integers(0, n_classes)
        if sizes[idx] > 1:
            sizes[idx] -= 1

    parts, labels = [], []
    for i, sz in enumerate(sizes):
        part = centers[i] + rng.normal(0, rng.uniform(0.8, 1.8), size=(sz, n_features))
        parts.append(part)
        labels.append(np.full(sz, i, dtype=int))

    X_full = np.vstack(parts)
    y_full = np.concatenate(labels)
    rng.shuffle(X_full)
    rng.shuffle(y_full)

    # Etiquetas faltantes (~15%)
    miss_mask = rng.random(y_full.shape) < 0.15
    y_with_nan = y_full.astype(float)
    y_with_nan[miss_mask] = np.nan

    # NaNs en X (~4%)
    nan_mask = rng.random(X_full.shape) < 0.04
    X_full[nan_mask] = np.nan

    df = pd.DataFrame(X_full, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y_with_nan

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "target_col": "target",
        "features": None,   # todas las numéricas excepto target
        "kernel": "rbf",
        "gamma": gamma,
        "unlabeled_fraction": unl_frac,
        "test_size": test_size,
        "random_state": random_state
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado
    # ---------------------------------------------------------
    # A. Selección de features (todas las numéricas excepto target)
    all_num = input_data["df"].select_dtypes(include=[np.number]).columns.tolist()
    if input_data["target_col"] in all_num:
        all_num.remove(input_data["target_col"])
    features = all_num

    # B. Usar solo filas con etiqueta conocida para split
    df_known = input_data["df"][input_data["df"][input_data["target_col"]].notna()].copy()
    X_known = df_known[features].to_numpy()
    y_known = df_known[input_data["target_col"]].astype(int).to_numpy()

    # C. Split estratificado
    X_train, X_test, y_train_full, y_test = train_test_split(
        X_known,
        y_known,
        test_size=float(input_data["test_size"]),
        random_state=int(input_data["random_state"]),
        stratify=y_known
    )

    # D. Ocultar fracción de etiquetas en train como -1
    rng2 = np.random.default_rng(int(input_data["random_state"]))
    n_train = len(y_train_full)
    n_unlab = int(np.clip(round(input_data["unlabeled_fraction"] * n_train), 0, n_train - 1))
    mask_unlab = np.zeros(n_train, dtype=bool)
    if n_unlab > 0:
        mask_unlab[rng2.choice(n_train, size=n_unlab, replace=False)] = True
    y_train_semi = y_train_full.copy()
    y_train_semi[mask_unlab] = -1

    # E. Preprocesamiento y LabelSpreading
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    ls      = LabelSpreading(kernel=input_data["kernel"], gamma=float(input_data["gamma"]))

    X_train_imp = imputer.fit_transform(X_train)
    X_train_sca = scaler.fit_transform(X_train_imp)
    ls.fit(X_train_sca, y_train_semi)

    X_test_imp = imputer.transform(X_test)
    X_test_sca = scaler.transform(X_test_imp)
    y_pred = ls.predict(X_test_sca).astype(int)

    acc = float(accuracy_score(y_test, y_pred))

    pipe = Pipeline(steps=[
        ("imputer", imputer),
        ("scaler", scaler),
        ("ls", ls)
    ])

    output_data = (y_test.astype(int), y_pred, acc, pipe)
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_propagar_etiquetas_label_spreading()
    y_test, y_pred, acc, _ = salida_esperada
    print("\n=== [LabelSpreading] INPUT ===")
    print({k: (f"DataFrame{entrada['df'].shape}" if k == "df" else v) for k, v in entrada.items()})
    print("=== [LabelSpreading] OUTPUT ESPERADO ===")
    print("y_test.shape:", y_test.shape, "| y_pred.shape:", y_pred.shape, "| accuracy:", round(acc, 4))