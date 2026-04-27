# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

RUTA_ENTRADA = os.path.join("DATA", "datos_brutos.csv")
RUTA_SALIDA = os.path.join("DATA", "dataset_final_maestro.csv")

INTERVALO_SEGUNDOS = 45


# =========================================================
# PARÁMETROS MODIFICABLES DE LA FUNCIÓN DE DAÑO
# =========================================================

# Temperatura de referencia normal
T_REF = 25.0

# Rangos usados para normalizar a [0, 1]
# P = Piezo_proxy normalizado
# T = desviación térmica normalizada
# A = aceleración dinámica normalizada
PIEZO_MIN = 0.0
PIEZO_MAX = 5000.0

ACCEL_MIN = 0.0
ACCEL_MAX = float(np.sqrt(3) * 20.0)  # ≈ 34.64 m/s²

TEMP_DEV_MIN = 0.0
TEMP_DEV_MAX = 40.0

# Pesos de tu función final:
# D = 0.45P + 0.10T + 0.45A
W_PIEZO = 0.45
W_TEMP = 0.10
W_ACCEL = 0.45

# Escala para pasar de daño instantáneo a daño acumulado.
# Esto NO cambia la función D.
# Solo controla cuánto sube el daño acumulado por cada fila del CSV.
DAMAGE_ACUM_SCALE = 0.10

# Umbrales de estado
UMBRAL_ATENCION = 30.0
UMBRAL_ALERTA = 60.0


# =========================================================
# PARÁMETROS DE LIMPIEZA
# =========================================================

FS = 2.0
FC = 0.3
AUTO_DETECTAR_FS = True

RANGOS = {
    "Accel_X (m/s^2)": (-20.0, 20.0),
    "Accel_Y (m/s^2)": (-20.0, 20.0),
    "Accel_Z (m/s^2)": (-20.0, 20.0),
    "Strain (με)": (-5000.0, 5000.0),
    "Temp (°C)": (-20.0, 400.0),
}

COL_VIBRACION = [
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)"
]


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def leer_csv_seguro(ruta_csv):
    return pd.read_csv(ruta_csv, on_bad_lines="skip")


def adaptar_columnas_si_hace_falta(df):
    """
    Permite que el script funcione con tu CSV del ESP32:

    timestamp_ms,temp_c,ax,ay,az,gx,gy,gz,piezo_raw

    y lo adapta al formato maestro usado por la función de daño:
    Temp (°C), Accel_X, Accel_Y, Accel_Z, Strain.
    """

    if "Temp (°C)" not in df.columns and "temp_c" in df.columns:
        df["Temp (°C)"] = pd.to_numeric(df["temp_c"], errors="coerce")

    if all(col in df.columns for col in ["ax", "ay", "az"]):
        accel_temp = df[["ax", "ay", "az"]].apply(pd.to_numeric, errors="coerce")

        max_abs = accel_temp.abs().max().max()

        # Si ax, ay, az parecen venir en g, los pasamos a m/s².
        # Si ya parecen venir en m/s², los dejamos igual.
        factor_conversion = 9.81 if max_abs <= 5 else 1.0

        if "Accel_X (m/s^2)" not in df.columns:
            df["Accel_X (m/s^2)"] = pd.to_numeric(df["ax"], errors="coerce") * factor_conversion

        if "Accel_Y (m/s^2)" not in df.columns:
            df["Accel_Y (m/s^2)"] = pd.to_numeric(df["ay"], errors="coerce") * factor_conversion

        if "Accel_Z (m/s^2)" not in df.columns:
            df["Accel_Z (m/s^2)"] = pd.to_numeric(df["az"], errors="coerce") * factor_conversion

    # En tu dataset de entrenamiento usabas Strain como proxy del piezo.
    # En el ESP32 real usamos piezo_raw como proxy equivalente.
    if "Strain (με)" not in df.columns and "piezo_raw" in df.columns:
        df["Strain (με)"] = pd.to_numeric(df["piezo_raw"], errors="coerce")

    if "Timestamp" not in df.columns and "timestamp_ms" in df.columns:
        ts = pd.to_numeric(df["timestamp_ms"], errors="coerce")

        if ts.notna().any():
            ts_min = ts.min()
            ts_max = ts.max()
            duracion_ms = ts_max - ts_min

            inicio = datetime.now() - timedelta(milliseconds=float(duracion_ms))
            df["Timestamp"] = inicio + pd.to_timedelta(ts - ts_min, unit="ms")

    return df


def inferir_fs(df):
    if not AUTO_DETECTAR_FS:
        return FS

    try:
        if "timestamp_ms" in df.columns:
            ts = pd.to_numeric(df["timestamp_ms"], errors="coerce").dropna().sort_values()
            diffs = ts.diff().dropna()
            diffs = diffs[diffs > 0]

            if len(diffs) > 0:
                dt_ms = diffs.median()
                if dt_ms > 0:
                    return 1000.0 / dt_ms

        if "Timestamp" in df.columns:
            ts = pd.to_datetime(df["Timestamp"], errors="coerce").dropna().sort_values()
            diffs = ts.diff().dropna().dt.total_seconds()
            diffs = diffs[diffs > 0]

            if len(diffs) > 0:
                dt_s = diffs.median()
                if dt_s > 0:
                    return 1.0 / dt_s

    except Exception:
        pass

    return FS


def filtro_fourier_pasa_bajos(serie, fs, fc):
    n = len(serie)

    if n < 2:
        return serie.to_numpy()

    serie_np = serie.to_numpy()

    espectro = fft(serie_np)
    frecuencias = fftfreq(n, d=1.0 / fs)

    espectro[np.abs(frecuencias) > fc] = 0

    senal_filtrada = np.real(ifft(espectro))
    return senal_filtrada


def norm_01(series, vmin, vmax):
    if vmax <= vmin:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return ((series - vmin) / (vmax - vmin)).clip(0, 1)


def calcular_estado(damage_acum_pct):
    if damage_acum_pct >= UMBRAL_ALERTA:
        return "Alerta"
    elif damage_acum_pct >= UMBRAL_ATENCION:
        return "Atención"
    else:
        return "Normal"


# =========================================================
# LIMPIEZA + NORMALIZACIÓN + FUNCIÓN DE DAÑO
# =========================================================

def limpiar_y_filtrar_maestro(ruta_in):
    print(f"\nIniciando limpieza maestra de: {ruta_in}")

    if not os.path.exists(ruta_in):
        raise FileNotFoundError(f"No existe el archivo de entrada: {ruta_in}")

    df = leer_csv_seguro(ruta_in)

    if df.empty:
        raise ValueError("El CSV de entrada está vacío.")

    # Adaptar columnas del ESP32 al formato maestro
    df = adaptar_columnas_si_hace_falta(df)

    # Timestamp, duplicados y orden
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        df = df.drop_duplicates(subset=["Timestamp"])
        df = df.sort_values("Timestamp")

    # Convertir a numérico y eliminar valores físicamente imposibles
    columnas_numericas_presentes = []

    for col, (vmin, vmax) in RANGOS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col].between(vmin, vmax))
            columnas_numericas_presentes.append(col)

    # Interpolación
    if columnas_numericas_presentes:
        df[columnas_numericas_presentes] = df[columnas_numericas_presentes].interpolate(
            method="linear",
            limit_direction="both"
        )

    if "Temp (°C)" in df.columns:
        df["Temp (°C)"] = df["Temp (°C)"].ffill().bfill()

        # Si el termopar no está conectado o no manda datos,
        # usamos temperatura neutra para que no se borren todas las filas.
        if df["Temp (°C)"].isna().all():
            df["Temp (°C)"] = T_REF

    # Fourier solo en vibración
    fs_real = inferir_fs(df)
    fc_real = min(FC, fs_real * 0.45)

    print(f"  → Frecuencia de muestreo usada: {fs_real:.3f} Hz")
    print(f"  → Aplicando Fourier pasa-bajos a {fc_real:.3f} Hz")

    for col in COL_VIBRACION:
        if col in df.columns:
            serie = pd.to_numeric(df[col], errors="coerce")
            serie = serie.ffill().bfill()

            if len(serie) >= 2:
                df[col] = filtro_fourier_pasa_bajos(serie, fs_real, fc_real)

    columnas_necesarias = [
        "Accel_X (m/s^2)",
        "Accel_Y (m/s^2)",
        "Accel_Z (m/s^2)",
        "Strain (με)",
        "Temp (°C)",
    ]

    faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if faltantes:
        raise ValueError(f"Faltan columnas necesarias en el CSV: {faltantes}")

    for col in columnas_necesarias:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[columnas_necesarias] = df[columnas_necesarias].interpolate(
        method="linear",
        limit_direction="both"
    )

    df[columnas_necesarias] = df[columnas_necesarias].ffill().bfill()

    if df["Temp (°C)"].isna().all():
        df["Temp (°C)"] = T_REF

    # =====================================================
    # VARIABLES DE ENTRADA DE LA FUNCIÓN DE DAÑO
    # =====================================================

    # Aceleración total bruta.
    # Esta incluye gravedad, por eso NO se usa directamente para daño.
    df["Accel_mag_raw"] = np.sqrt(
        df["Accel_X (m/s^2)"]**2 +
        df["Accel_Y (m/s^2)"]**2 +
        df["Accel_Z (m/s^2)"]**2
    )

    # Usamos las primeras muestras como referencia de reposo.
    # Así evitamos que la gravedad cuente como daño.
    n_base = min(30, len(df))

    accel_base = df["Accel_mag_raw"].head(n_base).median()
    piezo_base = df["Strain (με)"].head(n_base).median()

    if pd.isna(accel_base):
        accel_base = 0.0

    if pd.isna(piezo_base):
        piezo_base = 0.0

    # Aceleración dinámica: diferencia respecto al estado base.
    # Esta es la que se usa para el daño.
    df["Accel_mag"] = (df["Accel_mag_raw"] - accel_base).abs()

    # Piezo proxy corregido por offset inicial.
    df["Piezo_proxy"] = (df["Strain (με)"] - piezo_base).abs()

    # Desviación respecto a temperatura de referencia.
    df["Temp_dev"] = (df["Temp (°C)"] - T_REF).abs()

    # =====================================================
    # NORMALIZACIÓN DE P, T, A
    # =====================================================

    df["P"] = norm_01(df["Piezo_proxy"], PIEZO_MIN, PIEZO_MAX)
    df["T"] = norm_01(df["Temp_dev"], TEMP_DEV_MIN, TEMP_DEV_MAX)
    df["A"] = norm_01(df["Accel_mag"], ACCEL_MIN, ACCEL_MAX)

    # =====================================================
    # FUNCIÓN DE DAÑO FINAL
    # D = 0.45P + 0.10T + 0.45A
    # =====================================================

    df["D"] = (
        W_PIEZO * df["P"] +
        W_TEMP * df["T"] +
        W_ACCEL * df["A"]
    ).clip(0, 1)

    # Daño instantáneo en porcentaje
    df["Damage_inst_pct"] = 100 * df["D"]

    # Daño acumulado
    # Sale de acumular D en el tiempo.
    # DAMAGE_ACUM_SCALE controla cuánto aporta cada muestra al acumulado.
    df["Damage_acum_pct"] = (df["D"] * DAMAGE_ACUM_SCALE).cumsum().clip(0, 100)

    # Vida restante
    df["Vida_restante_pct"] = (100 - df["Damage_acum_pct"]).clip(0, 100)

    # Estado según daño acumulado
    df["Estado"] = df["Damage_acum_pct"].apply(calcular_estado)

    # Aliases por compatibilidad con dashboards que usen minúsculas
    df["damage_inst_pct"] = df["Damage_inst_pct"]
    df["damage_acum_pct"] = df["Damage_acum_pct"]
    df["vida_restante_pct"] = df["Vida_restante_pct"]
    df["estado"] = df["Estado"]

    # Orden final
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp")

    df = df.dropna().reset_index(drop=True)

    print(f"Proceso completado. Filas finales: {len(df)}")

    if len(df) > 0:
        print(f"D final instantáneo: {df['D'].iloc[-1]:.4f}")
        print(f"Daño instantáneo final: {df['Damage_inst_pct'].iloc[-1]:.2f}%")
        print(f"Daño acumulado final: {df['Damage_acum_pct'].iloc[-1]:.2f}%")
        print(f"Vida restante final: {df['Vida_restante_pct'].iloc[-1]:.2f}%")
        print(f"Estado final: {df['Estado'].iloc[-1]}")
        print(f"Base aceleración usada: {accel_base:.4f} m/s²")
        print(f"Base piezo usada: {piezo_base:.2f}")

    return df


# =========================================================
# EJECUCIÓN EN BUCLE
# =========================================================

if __name__ == "__main__":
    print("Monitor de limpieza + normalización + función de daño iniciado.")
    print(f"Entrada: {RUTA_ENTRADA}")
    print(f"Salida : {RUTA_SALIDA}")
    print(f"Actualización cada {INTERVALO_SEGUNDOS} s")

    while True:
        try:
            df_resultado = limpiar_y_filtrar_maestro(RUTA_ENTRADA)

            os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)

            df_resultado.to_csv(RUTA_SALIDA, index=False)

            print(f"CSV maestro actualizado correctamente en: {RUTA_SALIDA}")

        except Exception as e:
            print(f"Error durante la actualización: {e}")

        time.sleep(INTERVALO_SEGUNDOS)