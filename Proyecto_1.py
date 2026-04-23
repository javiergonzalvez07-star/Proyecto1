

import pandas as pd

CSV_PATH = "datos_sensores.csv"

# Rangos físicos válidos por sensor
RANGOS = {
    "accel_x":     (-20.0,  20.0),
    "accel_y":     (-20.0,  20.0),
    "accel_z":     (-20.0,  20.0),
    "gyro_x":      (-500.0, 500.0),
    "gyro_y":      (-500.0, 500.0),
    "gyro_z":      (-500.0, 500.0),
    "piezo_mv":    (0.0,    3300.0),
    "temperatura": (-20.0,  400.0),
}


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia un DataFrame de lecturas de sensores."""

    # 1. Parsear timestamp — filas con fecha inválida se descartan
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # 2. Convertir columnas numéricas — texto no convertible → NaN
    cols_num = ["accel_x", "accel_y", "accel_z",
                "gyro_x",  "gyro_y",  "gyro_z",
                "piezo_mv", "temperatura"]
    df[cols_num] = df[cols_num].apply(pd.to_numeric, errors="coerce")

    # 3. Eliminar duplicados por (timestamp, sensor_id)
    df = df.drop_duplicates(subset=["timestamp", "sensor_id"])

    # 4. Reemplazar valores fuera de rango físico con NaN
    for col, (vmin, vmax) in RANGOS.items():
        df[col] = df[col].where(df[col].between(vmin, vmax))

    # 5. Imputar NaN
    #    - señales de vibración/piezo → interpolación lineal
    #    - temperatura               → último valor conocido (ffill)
    cols_interp = ["accel_x", "accel_y", "accel_z",
                   "gyro_x",  "gyro_y",  "gyro_z", "piezo_mv"]
    df[cols_interp] = df[cols_interp].interpolate(method="linear",
                                                   limit_direction="both")
    df["temperatura"] = df["temperatura"].ffill().bfill()

    # 6. Descartar filas con NaN irresolubles y ordenar por tiempo
    df = df.dropna()
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def guardar(df: pd.DataFrame) -> None:
    """Añade el DataFrame al CSV (crea el archivo si no existe)."""
    cabecera = not pd.io.common.file_exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode="a", header=cabecera, index=False)
    print(f" {len(df)} fila(s) guardadas en {CSV_PATH}")


# ── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    datos = pd.DataFrame([
        # fila normal
        {"timestamp": "2026-03-25T10:00:00", "accel_x":  0.12, "accel_y": -0.05,
         "accel_z": 9.81, "gyro_x": 0.3, "gyro_y": -0.1, "gyro_z": 0.0,
         "piezo_mv": 145.0, "temperatura": 23.7, "sensor_id": "ESP32_01"},
        # NaN en accel_y
        {"timestamp": "2026-03-25T10:00:01", "accel_x":  0.10, "accel_y": None,
         "accel_z": 9.80, "gyro_x": 0.2, "gyro_y": -0.1, "gyro_z": 0.0,
         "piezo_mv": 148.0, "temperatura": 23.8, "sensor_id": "ESP32_01"},
        # outlier físico
        {"timestamp": "2026-03-25T10:00:02", "accel_x": 999.0, "accel_y": -0.04,
         "accel_z": 9.82, "gyro_x": 0.1, "gyro_y": -0.2, "gyro_z": 0.0,
         "piezo_mv": 150.0, "temperatura": 24.0, "sensor_id": "ESP32_01"},
        # timestamp inválido
        {"timestamp": "NO_ES_FECHA",          "accel_x":  0.09, "accel_y": -0.03,
         "accel_z": 9.80, "gyro_x": 0.1, "gyro_y":  0.0, "gyro_z": 0.0,
         "piezo_mv": 140.0, "temperatura": 23.5, "sensor_id": "ESP32_01"},
        # duplicado de la fila 1
        {"timestamp": "2026-03-25T10:00:00", "accel_x":  0.12, "accel_y": -0.05,
         "accel_z": 9.81, "gyro_x": 0.3, "gyro_y": -0.1, "gyro_z": 0.0,
         "piezo_mv": 145.0, "temperatura": 23.7, "sensor_id": "ESP32_01"},
        # fila normal final
        {"timestamp": "2026-03-25T10:00:05", "accel_x":  0.13, "accel_y": -0.07,
         "accel_z": 9.83, "gyro_x": 0.4, "gyro_y": -0.2, "gyro_z": 0.0,
         "piezo_mv": 152.0, "temperatura": 24.1, "sensor_id": "ESP32_01"},
    ])

    print(f"Filas brutas   : {len(datos)}")
    limpio = limpiar(datos)
    print(f"Filas limpias  : {len(limpio)}")
    print(limpio.to_string())

    guardar(limpio)
