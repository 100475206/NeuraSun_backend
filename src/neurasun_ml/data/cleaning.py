import pandas as pd

def clean_consumption_data(df_raw: pd.DataFrame, timezone: str = 'UTC'):
    """ Función para limpiar el dataset de consumo, cosas que hace:
        - Convertir la columna de timestamp a índice
        - Asegurar que está en la zona horaria correcta
        - Manejar valores faltantes
        - Renombrar columnas en caso de ser necesario
        - Filtrado de outliers (si aplica)
    """
    df = df_raw.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Eliminar filas sin timestamp válido
    df = df.dropna(subset=['timestamp'])

    df = df.set_index('timestamp')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if df.index.tz is None:
        df.index = df.index.tz_localize(
            timezone,
            ambiguous='NaT',
            nonexistent='shift_forward'
        )
    else:
        df.index = df.index.tz_convert(timezone)

    return df