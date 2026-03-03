import numpy as np
import pandas as pd
import random

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

def generar_caso_de_uso_clasificar_multietiqueta_hashing():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para un proceso que:
      - Construye X (texto) y Y (matriz binaria de etiquetas)
      - Split train/test
      - Pipeline: HashingVectorizer -> OneVsRest(LogisticRegression)
      - f1_micro en test
    OUTPUT: (Y_test, Y_pred, f1_micro, pipe)
    """

    # ---------------------------------------------------------
    # 1. Configuración aleatoria
    # ---------------------------------------------------------
    random_state = random.randint(0, 10**6)
    rng = np.random.default_rng(random_state)
    n_samples = random.randint(250, 500)
    n_features_hash = random.choice([2**12, 2**13, 2**14])
    test_size = random.choice([0.2, 0.25, 0.3])

    # ---------------------------------------------------------
    # 2. Generar datos (corpus + etiquetas)
    # ---------------------------------------------------------
    temas = {
        "deporte":    ["futbol", "baloncesto", "tenis", "gol", "liga", "equipo"],
        "politica":   ["eleccion", "gobierno", "congreso", "reforma", "debate"],
        "tecnologia": ["software", "nube", "ia", "modelo", "codigo", "algoritmo"],
        "economia":   ["mercado", "inversion", "inflacion", "credito", "empresa"]
    }
    etiquetas = list(temas.keys())

    textos = []
    Y_list = []
    for _ in range(n_samples):
        k = random.randint(1, 3)
        chosen = random.sample(etiquetas, k=k)
        tokens = []
        for t in chosen:
            tokens += random.choices(temas[t], k=random.randint(3, 6))
        tokens += random.choices(["analisis", "evento", "hoy", "datos", "reportaje", "global"], k=random.randint(2, 5))
        random.shuffle(tokens)
        textos.append(" ".join(tokens))
        Y_list.append([1 if tag in chosen else 0 for tag in etiquetas])

    df = pd.DataFrame({
        "texto": textos,
        **{f"etq_{e}": [row[i] for row in Y_list] for i, e in enumerate(etiquetas)}
    })
    label_cols = [f"etq_{e}" for e in etiquetas]

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "text_col": "texto",
        "label_cols": label_cols,
        "n_features": n_features_hash,
        "test_size": test_size,
        "random_state": random_state
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado
    # ---------------------------------------------------------
    X_text = input_data["df"][input_data["text_col"]].fillna("").astype(str).values
    Y = input_data["df"][input_data["label_cols"]].fillna(0).astype(int).values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_text, Y, test_size=float(input_data["test_size"]), random_state=int(input_data["random_state"])
    )

    hv = HashingVectorizer(n_features=int(input_data["n_features"]), alternate_sign=False, norm="l2")
    ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))

    X_train_vec = hv.transform(X_train)
    ovr.fit(X_train_vec, Y_train)

    X_test_vec = hv.transform(X_test)
    Y_pred = ovr.predict(X_test_vec)

    f1_micro = float(f1_score(Y_test, Y_pred, average="micro"))

    pipe = Pipeline(steps=[
        ("hash", hv),
        ("ovr", ovr)
    ])

    output_data = (np.asarray(Y_test, dtype=int), np.asarray(Y_pred, dtype=int), f1_micro, pipe)
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_clasificar_multietiqueta_hashing()
    Y_test, Y_pred, f1_micro, _ = salida_esperada
    print("\n=== [Multietiqueta] INPUT ===")
    print({k: (f"DataFrame{entrada['df'].shape}" if k == "df" else v) for k, v in entrada.items()})
    print("=== [Multietiqueta] OUTPUT ESPERADO ===")
    print("Y_test.shape:", Y_test.shape, "| Y_pred.shape:", Y_pred.shape, "| F1-micro:", round(f1_micro, 4))