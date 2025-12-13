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


    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Renombrado de variable consumo según entre
    df.rename(columns={
        'value': 'consumption',
        'cons': 'consumption',
        'consumo': 'consumption',
        'consumption': 'consumption'
    }, inplace=True)

    df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')

    # Manejo de NaNs en las columnas numéricas (por ahora eliminamos filas con NaN,
    # pero se puede mejorar con imputación)
    df = df.dropna(subset=['consumption'])

    # Filtrado de outliers (por ahora eliminamos valores negativos)
    df = df[df['consumption'] >= 0]
    df = df[df['consumption'] <= df['consumption'].quantile(0.99)]

    # Frecuencia temporal uniforme (15 minutos)
    df = df.asfreq('15min')
    df['consumption'] = df['consumption'].interpolate(method='time')


    return df