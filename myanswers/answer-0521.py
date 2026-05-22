import numpy as np
from sklearn.preprocessing import MinMaxScaler


def limpiar_y_escalar(df, columna):
    """Reemplaza outliers por p99 y aplica MinMaxScaler.

    Args:
        df (pandas.DataFrame): DataFrame con la columna a tratar.
        columna (str): nombre de la columna numérica.

    Returns:
        np.ndarray: array unidimensional con los valores escalados en [0,1].
    """
    df2 = df.copy()
    serie = df2[columna].dropna()
    if serie.empty:
        return np.array([], dtype=float)

    p99 = np.percentile(serie, 99)
    df2.loc[df2[columna] > p99, columna] = p99

    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df2[[columna]]).flatten()
    return arr


if __name__ == "__main__":
    # Este bloque es de prueba local: carga el generador si existe y compara
    try:
        import pickle
        inp = pickle.load(open('myanswers/cases/0521_input.pkl','rb'))
        expected = pickle.load(open('myanswers/cases/0521_output.pkl','rb'))
        res = limpiar_y_escalar(inp['df'], inp['columna'])
        print('Resultado shape:', res.shape)
    except Exception:
        pass
