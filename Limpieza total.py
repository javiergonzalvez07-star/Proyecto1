import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq


RUTA_ENTRADA = "building_health_monitoring_dataset__1_.csv"
RUTA_SALIDA  = "dataset_final_maestro.csv"

# Parámetros Fourier
FS = 1.0   # Frecuencia de muestreo (1 Hz)
FC = 0.3   # Frecuencia de corte (Hz)

RANGOS = {
    "Accel_X (m/s^2)": (-20.0, 20.0),
    "Accel_Y (m/s^2)": (-20.0, 20.0),
    "Accel_Z (m/s^2)": (-20.0, 20.0),
    "Strain (με)":      (-5000.0, 5000.0),
    "Temp (°C)":       (-20.0, 400.0),
}

COL_VIBRACION = ["Accel_X (m/s^2)", "Accel_Y (m/s^2)", "Accel_Z (m/s^2)"]

def filtro_fourier_pasa_bajos(señal, fs, fc):
    """Elimina el ruido de alta frecuencia usando Fourier."""
    n = len(señal)
    espectro = fft(señal.to_numpy())
    frecuencias = fftfreq(n, d=1.0 / fs)
    espectro[np.abs(frecuencias) > fc] = 0
    return np.real(ifft(espectro))

def limpiar_y_filtrar_maestro(ruta_in):
    print(f" Iniciando limpieza maestra de: {ruta_in}")
    df = pd.read_csv(ruta_in)

    # 1. Limpieza de Estructura y Duplicados
    # Usamos Timestamp si existe, si no, lo saltamos
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        df = df.drop_duplicates(subset=["Timestamp"])

    # 2. Convertir a numérico y Filtrar Rangos Imposibles (Outliers físicos)
    for col, (vmin, vmax) in RANGOS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Los valores fuera de rango se vuelven NaN para interpolarlos luego
            df[col] = df[col].where(df[col].between(vmin, vmax))

    # 3. Interpolación (Rellenar huecos de nulos y de los outliers eliminados)
    print("  → Rellenando huecos mediante interpolación lineal...")
    df = df.interpolate(method="linear", limit_direction="both")
    
    # Para temperatura usamos ffill/bfill (más seguro para cambios lentos)
    if "Temp (°C)" in df.columns:
        df["Temp (°C)"] = df["Temp (°C)"].ffill().bfill()

    # 4. Filtro de Fourier (Limpieza de ruido en vibración)
    print(f"  → Aplicando Fourier (Pasa-bajos {FC}Hz) a columnas de vibración...")
    for col in COL_VIBRACION:
        if col in df.columns:
            df[col] = filtro_fourier_pasa_bajos(df[col], FS, FC)

    # 5. Ordenar y finalizar
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp")
    
    df = df.dropna().reset_index(drop=True)
    print(f" Proceso completado. Filas finales: {len(df)}")
    return df

# ── Ejecución ──
if __name__ == "__main__":
    df_resultado = limpiar_y_filtrar_maestro(RUTA_ENTRADA)
    df_resultado.to_csv(RUTA_SALIDA, index=False)
    
    print("\nResumen del Dataset Maestro:")
    print(df_resultado.head(5))