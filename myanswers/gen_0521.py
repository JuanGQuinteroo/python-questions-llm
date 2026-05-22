import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generar_caso_de_uso():
    """
    Genera un caso de uso para el reto 'Limpieza de Outliers y Escalamiento'.
    Retorna una tupla: (dict_argumentos, objeto_resultado_esperado)
    """
    
    # --- 1. Creación de Datos Aleatorios (Dataset Sintético) ---
    np.random.seed(42)
    n_rows = 20
    col_name = 'voltaje'
    
    # Generamos datos y forzamos algunos picos absurdos (outliers)
    datos = np.random.uniform(10, 50, n_rows)
    datos[2] = 500.0
    datos[10] = 450.0
    
    df_input = pd.DataFrame({col_name: datos})

    # --- 2. Lógica de Referencia (Lo que debería hacer la función del usuario) ---
    df = df_input.copy()
    
    # Tratamiento de Outliers
    p99 = np.percentile(df[col_name], 99)
    df.loc[df[col_name] > p99, col_name] = p99
    
    # Escalamiento
    scaler = MinMaxScaler()
    resultado_array = scaler.fit_transform(df[[col_name]]).flatten()

    # --- 3. Construcción de la Tupla Final ---
    # El diccionario de argumentos contiene los parámetros que pide tu función
    argumentos = {
        "df": df_input,
        "columna": col_name
    }
    
    # El resultado esperado es directamente el array de numpy que devuelve la función
    resultado_esperado = {
        "array_escalado": resultado_array
    }

    return (argumentos, resultado_esperado)


# --- EJEMPLO DE USO ---
inputs, targets = generar_caso_de_uso()

print("--- ARGUMENTOS DE ENTRADA (Dataset) ---")
# Mostramos las primeras filas para ver el outlier en el índice 2
print(inputs['df'].head()) 
print(f"\nColumna objetivo: {inputs['columna']}")

print("\n" + "="*40 + "\n")

print("--- RESULTADO ESPERADO ---")
print("Primeros 5 elementos del array de numpy transformado:")
print(targets['array_escalado'][:5])
