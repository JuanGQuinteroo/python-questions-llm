import random

import numpy as np
import pandas as pd


def _resolver_construir_panel_diario(df, fecha_col, grupo_col, valor_col):
    trabajo = df.copy()
    trabajo[fecha_col] = pd.to_datetime(trabajo[fecha_col], errors="coerce")
    trabajo = trabajo.dropna(subset=[fecha_col, grupo_col, valor_col])

    if trabajo.empty:
        return pd.DataFrame(columns=[fecha_col, grupo_col, "valor_total"])

    agregado = trabajo.groupby([fecha_col, grupo_col], as_index=False)[valor_col].sum()

    fechas_completas = pd.date_range(
        start=agregado[fecha_col].min(),
        end=agregado[fecha_col].max(),
        freq="D",
    )
    grupos_validos = sorted(agregado[grupo_col].unique())

    indice_completo = pd.MultiIndex.from_product(
        [fechas_completas, grupos_validos], names=[fecha_col, grupo_col]
    )

    panel = (
        agregado.set_index([fecha_col, grupo_col])
        .reindex(indice_completo, fill_value=0.0)
        .reset_index()
    )

    panel = panel.rename(columns={valor_col: "valor_total"})
    panel = panel.sort_values([grupo_col, fecha_col]).reset_index(drop=True)
    panel["valor_total"] = panel["valor_total"].astype(float)
    return panel


def generar_caso_de_uso_construir_panel_diario():
    fecha_col = "fecha"
    grupo_col = "grupo"
    valor_col = "valor"

    n_grupos = random.randint(2, 4)
    grupos = [f"G{i}" for i in range(1, n_grupos + 1)]

    base = pd.Timestamp("2023-01-01") + pd.Timedelta(days=random.randint(0, 400))
    rango_dias = random.randint(8, 18)
    n_filas = random.randint(45, 100)

    filas = []
    for _ in range(n_filas):
        if random.random() < 0.12:
            fecha = "fecha_invalida"
        else:
            offset = random.randint(0, rango_dias - 1)
            fecha = (base + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")

        if random.random() < 0.08:
            grupo = None
        else:
            grupo = random.choice(grupos)

        if random.random() < 0.10:
            valor = np.nan
        else:
            valor = round(float(np.random.normal(loc=120.0, scale=55.0)), 2)

        filas.append({fecha_col: fecha, grupo_col: grupo, valor_col: valor})

    df = pd.DataFrame(filas)

    input_data = {
        "df": df.copy(),
        "fecha_col": fecha_col,
        "grupo_col": grupo_col,
        "valor_col": valor_col,
    }

    output_data = _resolver_construir_panel_diario(df, fecha_col, grupo_col, valor_col)
    return input_data, output_data
