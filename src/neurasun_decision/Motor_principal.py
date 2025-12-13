import pulp as pl
import pandas as pd
import numpy as np
from dataclasses import dataclass
import argparse
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

dataset_under = pd.read_csv(os.path.join(PROJECT_DIR, "datasets", "dataset_under.csv"))
dataset_over = pd.read_csv(os.path.join(PROJECT_DIR, "datasets", "dataset_over.csv"))


def debugging(data, p_charge, p_discharge, soc, p_grid):
    '''Función para debuggear los resultados del modelo'''
    horas = range(24)
    filas = []

    for h in horas:
        filas.append({
            'consumo': data["Consumos"][h],
            'generacion': data["Generaciones"][h],
            'precio': data["Precio"][h],
            'p_charge': p_charge[h].varValue,
            'p_discharge': p_discharge[h].varValue,
            'soc': soc[h].varValue,
            'p_grid': p_grid[h].varValue
        })

    df_debug = pd.DataFrame(filas)
    print(df_debug)
    return df_debug

def ahorro_NS_diario(data):
    '''Calculamos el ahorro diario con NeuraSun en una batería'''
    prob = pl.LpProblem("Optimizacion", pl.LpMinimize)
    CMAX = 215.0
    horas = range(24)

    # Potencia de carga y descarga separadas
    P_MAX = 100.0
    p_charge = pl.LpVariable.dicts("p_charge", horas, lowBound=0.0, upBound=P_MAX)
    p_discharge = pl.LpVariable.dicts("p_discharge", horas, lowBound=0.0, upBound=P_MAX)

    u = pl.LpVariable.dicts("u", horas, cat='Binary')

    for h in horas:
        prob += p_charge[h] <= P_MAX * u[h]
        prob += p_discharge[h] <= P_MAX * (1 - u[h])

    # SoC
    soc = pl.LpVariable.dicts("soc", horas, lowBound=0.2*CMAX, upBound=0.8*CMAX)
    prob += soc[0] == 0.2*CMAX

    # Importación de red
    p_grid = pl.LpVariable.dicts("p_grid", horas, lowBound=0.0)

    for h in horas:
        # Balance de potencia: red = consumo + carga batería - descarga batería - generación FV
        prob += p_grid[h] == data["Consumos"][h] + p_charge[h] - p_discharge[h] - data["Generaciones"][h]

    # Objetivo: minimizar coste de energía de red
    prob += pl.lpSum(data["Precio"][h] * p_grid[h] for h in horas)

    # Dinámica del SoC (Δt = 1h, sin pérdidas por simplicidad)
    for h in range(0, 23):
        prob += soc[h+1] == soc[h] + p_charge[h] - p_discharge[h]

    # Cerrar el día con mismo SoC (o al menos no regalar energía extra)
    #prob += soc[23] == soc[0]

    prob.solve(#pl.PULP_CBC_CMD(msg=1)
        )
    coste_NS = pl.value(prob.objective)
    df_debug = debugging(data, p_charge, p_discharge, soc, p_grid)
    return coste_NS

# Parámetros de la batería
BATTERY_CAPACITY_KWH = 215.0
BATTERY_SOC_MIN = 0.2 * BATTERY_CAPACITY_KWH
BATTERY_SOC_MAX = 0.8 * BATTERY_CAPACITY_KWH
BATTERY_P_CH_MAX_KW = 100.0
BATTERY_P_DIS_MAX_KW = 100.0
EFF_CH = 1.0
EFF_DIS = 1.0

TIME_STEP_HOURS = 1.0

#Cálculo de ahorros en factura sin batería, y con batería simple
@dataclass
class BatteryParams:
    capacity_kwh: float = BATTERY_CAPACITY_KWH
    soc_min: float = BATTERY_SOC_MIN
    soc_max: float = BATTERY_SOC_MAX
    p_ch_max_kw: float = BATTERY_P_CH_MAX_KW
    p_dis_max_kw: float = BATTERY_P_DIS_MAX_KW
    eff_ch: float = EFF_CH
    eff_dis: float = EFF_DIS
    dt_hours: float = TIME_STEP_HOURS


# Columnas esperadas en el CSV
COLUMN_CONSUMOS = "Consumos"
COLUMN_GENERACIONES = "Generaciones"
COLUMN_PRECIO = "Precio"


def calcular_coste_sin_bateria(df: pd.DataFrame) -> float:
    """Calcula el coste diario sin batería.

    El coste se calcula importando de red solo cuando el consumo
    supera a la generación (net_load > 0).
    """

    load = df[COLUMN_CONSUMOS].to_numpy(dtype=float)
    pv = df[COLUMN_GENERACIONES].to_numpy(dtype=float)
    price = df[COLUMN_PRECIO].to_numpy(dtype=float)

    net_load = load - pv
    grid_import = np.maximum(net_load, 0.0)
    cost = float(np.sum(grid_import * price))
    return cost


def calcular_coste_con_bateria(df: pd.DataFrame, params: BatteryParams) -> float:
    """Simula una batería simple y devuelve el coste diario asociado."""

    load = df[COLUMN_CONSUMOS].to_numpy(dtype=float)
    pv = df[COLUMN_GENERACIONES].to_numpy(dtype=float)
    price = df[COLUMN_PRECIO].to_numpy(dtype=float)

    hours = len(df)
    soc = params.capacity_kwh * 0.5  # SoC inicial al 50 %
    grid_import = np.zeros(hours)

    for t in range(hours):
        net_load = load[t] - pv[t]

        if net_load < 0:
            # Excedente FV disponible para cargar la batería.
            surplus = -net_load
            energy_space = params.soc_max - soc
            max_charge_energy = params.p_ch_max_kw * params.dt_hours
            energy_to_charge = min(surplus, energy_space, max_charge_energy)

            # Actualizar SoC respetando eficiencia de carga.
            soc += params.eff_ch * energy_to_charge
            soc = min(soc, params.soc_max)
            grid_import[t] = 0.0
        else:
            # Déficit: intentar cubrir con batería.
            deficit = net_load
            available_energy = soc - params.soc_min
            max_discharge_energy = params.p_dis_max_kw * params.dt_hours
            energy_discharge = min(deficit / params.eff_dis, available_energy, max_discharge_energy)

            energy_supplied = energy_discharge * params.eff_dis
            soc -= energy_discharge
            soc = max(soc, params.soc_min)

            deficit_after_battery = deficit - energy_supplied
            grid_import[t] = max(deficit_after_battery, 0.0)

        # Garantizar límites de SoC por seguridad numérica.
        soc = max(min(soc, params.soc_max), params.soc_min)

    cost = float(np.sum(grid_import * price))
    return cost

def leer_datos_csv(ruta_csv: str) -> pd.DataFrame:
    """Lee el CSV y valida que contiene las columnas esperadas."""

    df = pd.read_csv(ruta_csv)
    columnas_faltantes = {COLUMN_CONSUMOS, COLUMN_GENERACIONES, COLUMN_PRECIO} - set(df.columns)
    if columnas_faltantes:
        columnas_str = ", ".join(sorted(columnas_faltantes))
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {columnas_str}")

    return df

def plot_ahorro_acumulado(ahorro_batt_dia, ahorro_ns_dia, capex_bateria, anos):
    """Dibuja el ahorro acumulado a lo largo de los años para batería simple y con NeuraSun."""

    dias_totales = anos * 365

    dias = np.arange(1, dias_totales + 1)

    #Ahorro acumulado
    ahorro_acum_batt = ahorro_batt_dia * dias
    ahorro_acum_ns = ahorro_ns_dia * dias

    # Activar modo oscuro
    plt.style.use('dark_background')
    
    plt.figure()

    #Líneas de ahorro acumulado con sombra hacia abajo
    plt.plot(dias / 365, ahorro_acum_batt, label='Simple Battery', color='blue', linewidth=2)
    plt.fill_between(dias / 365, 0, ahorro_acum_batt, color='blue', alpha=0.3)
    
    plt.plot(dias / 365, ahorro_acum_ns, label='Battery + NeuraSun', color='gold', linewidth=2)
    plt.fill_between(dias / 365, 0, ahorro_acum_ns, color='gold', alpha=0.3)

    #Línea de CAPEX (sin sombra)
    plt.axhline(y=capex_bateria, color='red', linestyle='--', label='CAPEX Battery', linewidth=2)

    plt.xlabel('Years')
    plt.ylabel('Accumulated Savings (€)')
    plt.title('Accumulated Savings Over the Years')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcular ahorro diario al añadir una batería simple")
    parser.add_argument("--csv", default=os.path.join(PROJECT_DIR, "datasets", "dataset_under.csv"), help="Ruta del fichero CSV con 24 horas de datos")
    args = parser.parse_args()

    df = leer_datos_csv(args.csv)

    cost_no_batt = calcular_coste_sin_bateria(df)
    cost_batt = calcular_coste_con_bateria(df, BatteryParams())

    coste_con_ns = ahorro_NS_diario(df)
    ahorro_abs = cost_batt - coste_con_ns
    ahorro_rel = (ahorro_abs / cost_batt * 100) if cost_batt > 0 else 0.0
    ahorro_abs_anu = ahorro_abs * 365
    
    #porcentajes
    ns_vs_sin_batt = ((cost_no_batt-coste_con_ns)/cost_no_batt)*100
    retorno_bat_basic = (((cost_no_batt-cost_batt)*3650)/40000)*100 #meter aqui el coste bruto (40000)
    retorno_ns = (((cost_no_batt-coste_con_ns)*3650)/40000)*100 #meter aqui el coste bruto (40000)
    porc_NS_respecto_bat_basic = (ahorro_abs/(cost_no_batt-cost_batt))*100
    #valores de beneficio NeuraSun

    #Para la  gráfica de ahorros acumulados
    ahorro_batt = cost_no_batt - cost_batt
    ahorro_ns = cost_no_batt - coste_con_ns

    plot_ahorro_acumulado(ahorro_batt, ahorro_ns, 40000, 13)  # Suponiendo un CAPEX de 40000 € y 13 años

    print(f"Daily cost without battery: {cost_no_batt:.2f} €")
    print(f"Daily cost with simple battery: {cost_batt:.2f} €")
    print(f"Daily cost with battery + NeuraSun: {coste_con_ns:.2f} €")
    print(f"Extra daily savings (Battery + NeuraSun vs simple battery): {ahorro_abs:.2f} €")
    print(f"Extra annual savings (Battery + NeuraSun vs simple battery): {ahorro_abs_anu:.2f} €")
    print("================================")
    print("Percentages: ")
    print(f"Daily economic savings percentage NeuraSun vs Without Battery: {ns_vs_sin_batt:.2f} %")
    print(f"Relative savings with NeuraSun: {ahorro_rel:.2f} %")
    print(f"Percentage increase in savings (NeuraSun vs Simple Batt): {porc_NS_respecto_bat_basic:.2f} %")
    print("================================")
    print("Economic percentages: ")
    print(f"Basic battery return percentage (in 10 years): {retorno_bat_basic:.2f} %")
    print(f"NeuraSun battery return percentage (in 10 years): {retorno_ns:.2f} %")
    

#Comentar para solo ver pruebas
if __name__ == "__main__":
    main()