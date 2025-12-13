import pandas as pd
from pathlib import Path

Precios = "datasets/raw_datasets/prices_prediction_24h.csv"
Consumos = "datasets/raw_datasets/consumption_24h.csv"
Generacion = "datasets/raw_datasets/solar_prediction_24h.csv"

def readCSVs():

    df = pd.read_csv(Precios, sep=";")
    
    df = df.drop(labels=["Periodo", "Variación", "Hora"], axis=1)
    df["Precio"] = df["Precio"].str[:6].str.replace(",", ".").astype(float)

    consumos = pd.read_csv(Consumos)

    df["Consumos"] = consumos["consumption"]
    
    generaciones = pd.read_csv(Generacion)
    df["Generaciones"] = generaciones["SolarGeneration"]

    df.to_csv("datasets/info.csv")

readCSVs()
