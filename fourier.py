"""
LIMPIEZA AUTOMÁTICA DE SEÑALES DE VIBRACIÓN CON FOURIER
========================================================
Dataset: building_health_monitoring_dataset.csv
Señales tratadas: Accel_X, Accel_Y, Accel_Z (equivalente al piezoeléctrico)

Pasos:
  1. Carga y rellena valores nulos (interpolación lineal)
  2. Aplica filtro pasa-bajos con Fourier (elimina ruido de alta frecuencia)
  3. Guarda el CSV limpio listo para usar en el modelo
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq

# ─────────────────────────────────────────────
# PARÁMETROS — ajusta aquí si hace falta
# ─────────────────────────────────────────────
RUTA_ENTRADA  = r"building_health_monitoring_dataset__1_.csv"
RUTA_SALIDA   = r"/mnt/user-data/outputs/dataset_limpio.csv"

FRECUENCIA_MUESTREO = 1.0   # Hz — 1 muestra por segundo (según el Timestamp del CSV) / Dependiendo del sensor 
FRECUENCIA_CORTE    = 0.3   # Hz — frecuencias por encima de esto se eliminan como ruido
                             # Para daño estructural, lo relevante está en baja frecuencia.
                             # Si ves que se pierde demasiada señal, súbelo a 0.4 o 0.45

COLUMNAS_VIBRACION = ["Accel_X (m/s^2)", "Accel_Y (m/s^2)", "Accel_Z (m/s^2)"]
# ─────────────────────────────────────────────


def rellenar_nulos(df, columnas):
    """Rellena huecos con interpolación lineal y, si quedan extremos, los copia del vecino más cercano."""
    df[columnas] = df[columnas].interpolate(method="linear").ffill().bfill()
    return df


def filtro_fourier_pasa_bajos(señal, fs, fc):
    """
    Filtra una señal 1D con Fourier.
    - fs: frecuencia de muestreo (Hz)
    - fc: frecuencia de corte (Hz) — todo lo que esté por encima se elimina
    Devuelve la señal limpia (misma longitud).
    """
    n = len(señal)

    # 1. Transformada de Fourier → descomponemos la señal en frecuencias
    espectro = fft(señal)

    # 2. Calculamos a qué frecuencia corresponde cada componente
    frecuencias = fftfreq(n, d=1.0 / fs)

    # 3. Ponemos a cero las frecuencias por encima del corte (el "ruido")
    espectro[np.abs(frecuencias) > fc] = 0

    # 4. Transformada inversa → volvemos al dominio del tiempo, ya limpio
    señal_limpia = np.real(ifft(espectro))

    return señal_limpia


def limpiar_dataset(ruta_entrada, ruta_salida, columnas, fs, fc):
    print(f"Cargando: {ruta_entrada}")
    df = pd.read_csv(ruta_entrada)
    print(f"  → {len(df)} filas, {df.isnull().sum().sum()} valores nulos en total")

    # Paso 1: rellenar nulos
    df = rellenar_nulos(df, columnas)
    nulos_tras_relleno = df[columnas].isnull().sum().sum()
    print(f"  → Nulos tras interpolación: {nulos_tras_relleno}")

    # Paso 2: aplicar filtro Fourier a cada columna de vibración
    print(f"\nAplicando filtro pasa-bajos (fc = {fc} Hz) a: {columnas}")
    for col in columnas:
        señal_original = df[col].to_numpy(dtype=float)
        df[col] = filtro_fourier_pasa_bajos(señal_original, fs=fs, fc=fc)
        print(f"  ✓ {col} — listo")

    # Paso 3: guardar
    df.to_csv(ruta_salida, index=False)
    print(f"\n✅ Dataset limpio guardado en: {ruta_salida}")
    print(f"   Filas: {len(df)} | Columnas: {list(df.columns)}")
    return df


# ─── Ejecución ───────────────────────────────
if __name__ == "__main__":
    df_limpio = limpiar_dataset(
        ruta_entrada=RUTA_ENTRADA,
        ruta_salida=RUTA_SALIDA,
        columnas=COLUMNAS_VIBRACION,
        fs=FRECUENCIA_MUESTREO,
        fc=FRECUENCIA_CORTE,
    )

    # Vista rápida del resultado
    print("\nPrimeras filas del dataset limpio:")
    print(df_limpio[["Timestamp"] + COLUMNAS_VIBRACION + ["Condition Label"]].head(8).to_string(index=False))