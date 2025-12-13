import pandas as pd

def load_consumption_raw(csv_path: str):
    """ Carga el dataset crudo de consumo desde un CSV"""
    df = pd.read_csv(csv_path)
    return df