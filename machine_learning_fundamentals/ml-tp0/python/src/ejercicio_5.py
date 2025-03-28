import pandas as pd

# Lectura de CSV
def read_data(ruta_archivo):
    df = pd.read_csv(ruta_archivo)
    columns = df.columns.tolist() 
    num_rows = len(df)
    return columns, num_rows

# Modificación de datos
def modify_data(ruta_archivo):
    df = pd.read_csv(ruta_archivo) 
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["mes"] = df["fecha"].apply(lambda x: x.month)
    return df

# Agrupación
def group_data(ruta_archivo):
    df = pd.read_csv(ruta_archivo)
    df = df.groupby("tipo").sum(numeric_only=True)
    return df