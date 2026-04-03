import random

import numpy as np
import pandas as pd


def _resolver_rankear_clientes_ventana_movil(
    df, cliente_col, fecha_col, monto_col, ventana
):
    trabajo = df.copy()
    trabajo[fecha_col] = pd.to_datetime(trabajo[fecha_col], errors="coerce")
    trabajo = trabajo.dropna(subset=[cliente_col, fecha_col, monto_col])

    if trabajo.empty:
        return pd.DataFrame(columns=[fecha_col, cliente_col, "suma_movil"])

    agregado = (
        trabajo.groupby([cliente_col, fecha_col], as_index=False)[monto_col]
        .sum()
        .sort_values([cliente_col, fecha_col])
        .reset_index(drop=True)
    )

    agregado["suma_movil"] = agregado.groupby(cliente_col)[monto_col].transform(
        lambda serie: serie.rolling(window=ventana, min_periods=1).sum()
    )

    ranking = agregado.sort_values(
        [fecha_col, "suma_movil", cliente_col],
        ascending=[True, False, True],
    )

    top_por_fecha = (
        ranking.groupby(fecha_col, as_index=False)
        .first()[[fecha_col, cliente_col, "suma_movil"]]
        .sort_values(fecha_col)
        .reset_index(drop=True)
    )

    top_por_fecha["suma_movil"] = top_por_fecha["suma_movil"].astype(float)
    return top_por_fecha


def generar_caso_de_uso_rankear_clientes_ventana_movil():
    cliente_col = "cliente"
    fecha_col = "fecha"
    monto_col = "monto"

    clientes = [f"C{i:02d}" for i in range(1, random.randint(4, 7))]
    ventana = random.randint(2, 5)

    base = pd.Timestamp("2022-01-01") + pd.Timedelta(days=random.randint(0, 600))
    rango_dias = random.randint(10, 24)
    n_filas = random.randint(70, 150)

    filas = []
    for _ in range(n_filas):
        if random.random() < 0.10:
            fecha = "no_fecha"
        else:
            offset = random.randint(0, rango_dias - 1)
            fecha = (base + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")

        if random.random() < 0.07:
            cliente = None
        else:
            cliente = random.choice(clientes)

        if random.random() < 0.10:
            monto = np.nan
        else:
            monto = round(float(np.random.normal(loc=90.0, scale=40.0)), 2)

        filas.append({cliente_col: cliente, fecha_col: fecha, monto_col: monto})

    df = pd.DataFrame(filas)

    input_data = {
        "df": df.copy(),
        "cliente_col": cliente_col,
        "fecha_col": fecha_col,
        "monto_col": monto_col,
        "ventana": ventana,
    }

    output_data = _resolver_rankear_clientes_ventana_movil(
        df, cliente_col, fecha_col, monto_col, ventana
    )
    return input_data, output_data
