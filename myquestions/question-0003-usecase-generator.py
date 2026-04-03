import random

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _resolver_proyectar_y_reconstruir_pca(X, n_componentes):
    scaler = StandardScaler()
    X_escalada = scaler.fit_transform(X)

    pca = PCA(n_components=n_componentes, svd_solver="full")
    X_proyectada = pca.fit_transform(X_escalada)

    X_reconstruida_escalada = pca.inverse_transform(X_proyectada)
    X_reconstruida = scaler.inverse_transform(X_reconstruida_escalada)

    varianza_explicada = pca.explained_variance_ratio_
    return X_proyectada, X_reconstruida, varianza_explicada


def generar_caso_de_uso_proyectar_y_reconstruir_pca():
    n_muestras = random.randint(35, 90)
    n_features = random.randint(4, 8)
    n_componentes = random.randint(2, n_features - 1)

    dim_latente = random.randint(2, min(4, n_features))
    latente = np.random.normal(size=(n_muestras, dim_latente))
    pesos = np.random.uniform(-2.5, 2.5, size=(dim_latente, n_features))
    ruido = np.random.normal(
        loc=0.0,
        scale=random.uniform(0.05, 0.35),
        size=(n_muestras, n_features),
    )

    X = latente @ pesos + ruido

    input_data = {
        "X": X.copy(),
        "n_componentes": n_componentes,
    }

    output_data = _resolver_proyectar_y_reconstruir_pca(X, n_componentes)
    return input_data, output_data
