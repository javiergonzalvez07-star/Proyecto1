import os
import time
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq


# =========================
# RUTAS
# =========================
RUTA_ENTRADA = "datos_brutos.csv"
RUTA_SALIDA = "dataset_final_maestro.csv"

# =========================
# PARÁMETROS FOURIER
# OJO: FS debe coincidir con tu frecuencia real de muestreo
# =========================
FS = 1.0   # Hz
FC = 0.3   # Hz

# =========================
# RANGOS FÍSICOS ADMISIBLES
# =========================
RANGOS = {
    "Accel_X (m/s^2)": (-20.0, 20.0),
    "Accel_Y (m/s^2)": (-20.0, 20.0),
    "Accel_Z (m/s^2)": (-20.0, 20.0),
    "Strain (με)": (-5000.0, 5000.0),
    "Temp (°C)": (-20.0, 400.0),
}

COL_VIBRACION = ["Accel_X (m/s^2)", "Accel_Y (m/s^2)", "Accel_Z (m/s^2)"]

# =========================
# PARÁMETROS DE NORMALIZACIÓN
# Ajusta estos valores cuando saques los definitivos del notebook
# =========================
T_REF = 25.0  # Temperatura de referencia "normal"

NORM_PARAMS = {
    # Piezo_proxy = abs(Strain)
    "Piezo_proxy": {"min": 0.0, "max": 5000.0},

    # Accel_mag = sqrt(ax^2 + ay^2 + az^2)
    # Si cada eje está en [-20, 20], el máximo teórico es sqrt(3)*20 ≈ 34.64
    "Accel_mag": {"min": 0.0, "max": float(np.sqrt(3) * 20.0)},

    # Temp_dev = abs(Temp - T_REF)
    # Ajusta este max cuando calibres con tus datos reales
    "Temp_dev": {"min": 0.0, "max": 40.0},
}

# =========================
# TIEMPO DE ACTUALIZACIÓN
# =========================
INTERVALO_SEGUNDOS = 45


def filtro_fourier_pasa_bajos(serie, fs, fc):
    """Elimina ruido de alta frecuencia usando FFT."""
    n = len(serie)
    if n < 2:
        return serie.to_numpy()

    espectro = fft(serie.to_numpy())
    frecuencias = fftfreq(n, d=1.0 / fs)
    espectro[np.abs(frecuencias) > fc] = 0

    señal_filtrada = np.real(ifft(espectro))
    return señal_filtrada


def norm_01(series, vmin, vmax):
    """Normaliza a [0,1] y recorta fuera de rango."""
    if vmax <= vmin:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - vmin) / (vmax - vmin)).clip(0, 1)


def leer_csv_seguro(ruta_csv):
    """
    Lee el CSV tolerando líneas incompletas.
    Útil si el archivo se está actualizando mientras Python lo lee.
    """
    return pd.read_csv(ruta_csv, on_bad_lines="skip")


def limpiar_y_filtrar_maestro(ruta_in):
    print(f"\nIniciando limpieza maestra de: {ruta_in}")

    if not os.path.exists(ruta_in):
        raise FileNotFoundError(f"No existe el archivo de entrada: {ruta_in}")

    df = leer_csv_seguro(ruta_in)

    if df.empty:
        raise ValueError("El CSV de entrada está vacío.")

    # =========================
    # 1. Timestamp, duplicados y orden básico
    # =========================
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        df = df.drop_duplicates(subset=["Timestamp"])

    # =========================
    # 2. Convertir a numérico y filtrar outliers físicos
    # =========================
    columnas_numericas_presentes = []

    for col, (vmin, vmax) in RANGOS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col].between(vmin, vmax))
            columnas_numericas_presentes.append(col)

    # =========================
    # 3. Interpolación solo en columnas numéricas
    # =========================
    if columnas_numericas_presentes:
        print("  → Rellenando huecos mediante interpolación lineal...")
        df[columnas_numericas_presentes] = df[columnas_numericas_presentes].interpolate(
            method="linear",
            limit_direction="both"
        )

    # Para temperatura, refuerzo con forward/backward fill
    if "Temp (°C)" in df.columns:
        df["Temp (°C)"] = df["Temp (°C)"].ffill().bfill()

    # =========================
    # 4. Fourier solo en vibración
    # =========================
    print(f"  → Aplicando Fourier (pasa-bajos {FC} Hz) a columnas de vibración...")
    for col in COL_VIBRACION:
        if col in df.columns:
            serie = pd.to_numeric(df[col], errors="coerce")
            serie = serie.ffill().bfill()
            if len(serie) >= 2:
                df[col] = filtro_fourier_pasa_bajos(serie, FS, FC)


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

    df["Accel_mag"] = np.sqrt(
        df["Accel_X (m/s^2)"]**2 +
        df["Accel_Y (m/s^2)"]**2 +
        df["Accel_Z (m/s^2)"]**2
    )

    df["Piezo_proxy"] = df["Strain (με)"].abs()

    df["Temp_dev"] = (df["Temp (°C)"] - T_REF).abs()

# NOrmalizar a 0s y 1s
    df["P_norm"] = norm_01(
        df["Piezo_proxy"],
        NORM_PARAMS["Piezo_proxy"]["min"],
        NORM_PARAMS["Piezo_proxy"]["max"]
    )

    df["A_norm"] = norm_01(
        df["Accel_mag"],
        NORM_PARAMS["Accel_mag"]["min"],
        NORM_PARAMS["Accel_mag"]["max"]
    )

    df["T_norm"] = norm_01(
        df["Temp_dev"],
        NORM_PARAMS["Temp_dev"]["min"],
        NORM_PARAMS["Temp_dev"]["max"]
    )

# Orden y limpieza final
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp")

    df = df.dropna().reset_index(drop=True)

    print(f"Proceso completado. Filas finales: {len(df)}")
    return df


# ejecutar en bucle
if __name__ == "__main__":
    print("Monitor de limpieza + normalización iniciado.")
    print(f"Entrada: {RUTA_ENTRADA}")
    print(f"Salida : {RUTA_SALIDA}")
    print(f"Actualización cada {INTERVALO_SEGUNDOS} s")

    while True:
        try:
            df_resultado = limpiar_y_filtrar_maestro(RUTA_ENTRADA)
            df_resultado.to_csv(RUTA_SALIDA, index=False)
            print("CSV maestro actualizado correctamente.")

        except Exception as e:
            print(f"Error durante la actualización: {e}")

        time.sleep(INTERVALO_SEGUNDOS)