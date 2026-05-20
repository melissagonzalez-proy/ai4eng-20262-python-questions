import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def seleccionar_mejores_variables(X, y):
    """
    Selecciona las 5 variables más relevantes usando SelectKBest con f_classif.
    
    Parámetros:
    -----------
    X : pandas.DataFrame o numpy.ndarray
        Matriz de características (variables predictoras)
    y : pandas.Series o numpy.ndarray
        Variable objetivo binaria (riesgo financiero)
    
    Retorna:
    --------
    dict
        Diccionario con dos claves:
        - 'variables_seleccionadas': lista con los nombres de las 5 mejores variables
        - 'X_reducido': DataFrame con solo las 5 variables seleccionadas
    """
    # Aplicar SelectKBest con f_classif para seleccionar las 5 mejores variables
    selector = SelectKBest(score_func=f_classif, k=5)
    X_reducido_array = selector.fit_transform(X, y)
    
    # Obtener los índices de las variables seleccionadas
    indices_seleccionados = selector.get_support(indices=True)
    
    # Obtener los nombres de las variables seleccionadas
    if hasattr(X, 'columns'):
        # Si X es DataFrame, usar los nombres de las columnas
        variables_seleccionadas = X.columns[indices_seleccionados].tolist()
        # Crear DataFrame reducido con los nombres originales
        X_reducido = pd.DataFrame(X_reducido_array, 
                                  columns=variables_seleccionadas,
                                  index=X.index if hasattr(X, 'index') else None)
    else:
        # Si X es un array, usar nombres genéricos
        variables_seleccionadas = [f"variable_{i}" for i in indices_seleccionados]
        X_reducido = pd.DataFrame(X_reducido_array, 
                                  columns=variables_seleccionadas)
    
    return {
        "variables_seleccionadas": variables_seleccionadas,
        "X_reducido": X_reducido
    }