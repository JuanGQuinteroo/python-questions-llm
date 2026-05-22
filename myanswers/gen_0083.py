import numpy as np
import random
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsRegressor


def generar_caso_de_uso_entrenar_brazo_robotico():
    """
    Genera un caso de uso aleatorio para la función entrenar_brazo_robotico.
    Retorna un diccionario 'input' con X e y, y el 'output' esperado (Pipeline entrenado).
    """
    # 1. Definir dimensiones aleatorias para el dataset
    n_samples = random.randint(150, 500)
    n_features = random.randint(4, 10)

    # 2. Generar datos base usando sklearn (multiobjetivo: n_targets=2 para X e Y)
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=2,
        noise=0.5,
        random_state=random.randint(1, 1000)
    )

    # 3. Inyectar ruido extremo (Outliers) en los sensores (X)
    # Seleccionamos un 10% de los datos para corromperlos
    n_outliers = int(n_samples * 0.1)
    for _ in range(n_outliers):
        row = random.randint(0, n_samples - 1)
        col = random.randint(0, n_features - 1)
        # Multiplicar por un factor gigante para simular un fallo del sensor
        X[row, col] = X[row, col] * random.choice([20, -20, 50, -50])

    input_dict = {
        'X': X,
        'y': y
    }

    # 4. Calcular el output esperado (solución)
    # Construcción del Pipeline exigido en el enunciado
    pipeline_esperado = Pipeline([
        ('transformer', QuantileTransformer(
            output_distribution='normal', random_state=42)),
        ('model', KNeighborsRegressor(n_neighbors=5))
    ])

    # Entrenar el pipeline con los datos generados
    pipeline_esperado.fit(X, y)

    return input_dict, pipeline_esperado


# --- Ejemplo de uso ---
if __name__ == "__main__":
    inputs, output_esperado = generar_caso_de_uso_entrenar_brazo_robotico()
    print("Forma de X:", inputs['X'].shape)
    print("Forma de y:", inputs['y'].shape)
    print("Pasos del Pipeline:", output_esperado.named_steps)
