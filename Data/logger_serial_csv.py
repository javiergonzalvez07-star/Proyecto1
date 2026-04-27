import serial
import csv
import os
from datetime import datetime

# =========================
# CONFIGURACIÓN
# =========================

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

DATA_FOLDER = "DATA"
CSV_FILE = "datos_brutos.csv"

CSV_PATH = os.path.join(DATA_FOLDER, CSV_FILE)

EXPECTED_COLUMNS = [
    "timestamp_ms",
    "temp_c",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "piezo_raw"
]


# =========================
# PREPARAR CSV
# =========================

os.makedirs(DATA_FOLDER, exist_ok=True)

file_exists = os.path.exists(CSV_PATH)

csv_file = open(CSV_PATH, mode="a", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

if not file_exists or os.path.getsize(CSV_PATH) == 0:
    writer.writerow(EXPECTED_COLUMNS)


# =========================
# LEER SERIAL Y GUARDAR
# =========================

print(f"Leyendo datos desde {SERIAL_PORT} a {BAUD_RATE} baudios...")
print(f"Guardando en: {CSV_PATH}")
print("Pulsa CTRL + C para parar.\n")

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        # Ignorar mensajes informativos del ESP32
        if line.startswith("#"):
            print(line)
            continue

        # Ignorar cabecera enviada por Arduino
        if line.startswith("timestamp_ms"):
            continue

        parts = line.split(",")

        # Solo guardar líneas que tengan las 9 columnas esperadas
        if len(parts) != len(EXPECTED_COLUMNS):
            print(f"Línea ignorada: {line}")
            continue

        writer.writerow(parts)
        csv_file.flush()

        print(f"Guardado: {line}")

except KeyboardInterrupt:
    print("\nLectura detenida por el usuario.")

finally:
    ser.close()
    csv_file.close()
    print("Puerto serie cerrado.")