import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsRegressor


def entrenar_brazo_robotico(X, y):
    """Construye y entrena pipeline: QuantileTransformer -> KNeighborsRegressor.

    Args:
        X (np.ndarray): matriz de características.
        y (np.ndarray): vector objetivo con dos columnas.

    Returns:
        Pipeline: pipeline entrenado.
    """
    pipeline = Pipeline([
        ('transformer', QuantileTransformer(output_distribution='normal', random_state=42)),
        ('model', KNeighborsRegressor(n_neighbors=5)),
    ])
    pipeline.fit(X, y)
    return pipeline


if __name__ == '__main__':
    try:
        import pickle
        inp = pickle.load(open('myanswers/cases/0083_input.pkl','rb'))
        pipeline = entrenar_brazo_robotico(inp['X'], inp['y'])
        print('Pipeline steps:', pipeline.named_steps.keys())
    except Exception:
        pass
