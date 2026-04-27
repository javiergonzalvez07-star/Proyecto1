# simulador_demo.py

from pathlib import Path
from datetime import datetime, timedelta
import csv
import time
import math
import random
import shutil
import unicodedata


# ============================================================
# CONFIGURACIÓN PRINCIPAL
# ============================================================

CSV_PATH = Path("DATA") / "dataset_final_maestro.csv"

# Duración real de la demo
DEMO_DURATION_SECONDS = 120        # 2 minutos reales

# Tiempo simulado dentro de la demo
SIMULATED_HOURS = 24               # 24 h simuladas en 2 minutos

# Cada cuánto se escribe una fila nueva
UPDATE_EVERY_SECONDS = 1.0

# Cada cuánto pasa un tren en tiempo real de demo
TRAIN_INTERVAL_RANGE_SECONDS = (8, 14)

# Duración visible del paso de un tren
TRAIN_DURATION_SECONDS = 3.0

# Daño inicial normalizado: 0 = sin daño, 1 = daño máximo
INITIAL_DAMAGE = 0.08

# Si True, borra el CSV completo al empezar la demo.
# OJO: déjalo en False para no perder tus datos reales.
CLEAR_CSV_AT_START = False

# Si True, elimina filas antiguas de demos anteriores, pero mantiene datos reales.
REMOVE_PREVIOUS_DEMO_ROWS = True

SENSOR_ID = "DEMO_SENSOR_01"
DEMO_SOURCE = "synthetic_demo"


# ============================================================
# COLUMNAS DE DEMO
# ============================================================

DEMO_COLUMNS = [
    "timestamp",
    "simulated_time",
    "simulated_hour",
    "source",
    "sensor_id",
    "event_type",
    "train_detected",
    "temperature_C",
    "accel_x",
    "accel_y",
    "accel_z",
    "accel_magnitude",
    "vibration",
    "piezo",
    "damage",
    "life_percent",
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalize_name(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace(" ", "_").replace("-", "_")
    return text


def backup_csv(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        backup_name = path.with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        shutil.copy(path, backup_name)
        print(f"[OK] Backup creado: {backup_name}")


def ensure_csv_schema(path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)

    if CLEAR_CSV_AT_START:
        backup_csv(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DEMO_COLUMNS)
            writer.writeheader()
        return DEMO_COLUMNS

    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DEMO_COLUMNS)
            writer.writeheader()
        return DEMO_COLUMNS

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_columns = reader.fieldnames or []
        rows = list(reader)

    final_columns = existing_columns[:]

    for col in DEMO_COLUMNS:
        if col not in final_columns:
            final_columns.append(col)

    must_rewrite = final_columns != existing_columns

    if REMOVE_PREVIOUS_DEMO_ROWS and "source" in final_columns:
        original_len = len(rows)
        rows = [row for row in rows if row.get("source") != DEMO_SOURCE]
        if len(rows) != original_len:
            must_rewrite = True

    if must_rewrite:
        backup_csv(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_columns)
            writer.writeheader()
            for row in rows:
                clean_row = {col: row.get(col, "") for col in final_columns}
                writer.writerow(clean_row)

    return final_columns


def value_for_existing_column(column: str, data: dict):
    """
    Esta función intenta rellenar también columnas que ya existan en tu CSV,
    aunque tengan nombres ligeramente distintos.
    """
    name = normalize_name(column)

    if name in ["timestamp", "date", "datetime", "time_stamp"]:
        return data["timestamp"]

    if name in ["simulated_time", "hora_simulada"]:
        return data["simulated_time"]

    if name in ["simulated_hour", "hora", "hour"]:
        return data["simulated_hour"]

    if name in ["source", "origen"]:
        return data["source"]

    if name in ["sensor_id", "sensor", "id_sensor"]:
        return data["sensor_id"]

    if name in ["event_type", "evento", "estado"]:
        return data["event_type"]

    if name in ["train_detected", "tren_detectado"]:
        return data["train_detected"]

    if "temp" in name or "temperature" in name or "temperatura" in name:
        return data["temperature_C"]

    if name in ["accel_x", "acc_x", "ax", "acceleration_x", "aceleracion_x"]:
        return data["accel_x"]

    if name in ["accel_y", "acc_y", "ay", "acceleration_y", "aceleracion_y"]:
        return data["accel_y"]

    if name in ["accel_z", "acc_z", "az", "acceleration_z", "aceleracion_z"]:
        return data["accel_z"]

    if (
        "accel_magnitude" in name
        or "acceleration_magnitude" in name
        or "acc_magnitude" in name
        or "modulo_aceleracion" in name
        or name == "acceleration"
        or name == "aceleracion"
    ):
        return data["accel_magnitude"]

    if "vibration" in name or "vibracion" in name or "vibracion" in name:
        return data["vibration"]

    if "piezo" in name or "strain" in name:
        return data["piezo"]

    if "damage" in name or "dano" in name or "danio" in name:
        return data["damage"]

    if "life" in name or "vida" in name or "health" in name:
        return data["life_percent"]

    if name in ["condition_label", "label", "class"]:
        return "train_event" if data["train_detected"] == 1 else "normal"

    return ""


def append_row(path: Path, fieldnames: list[str], data: dict) -> None:
    row = {}

    for col in fieldnames:
        if col in data:
            row[col] = data[col]
        else:
            row[col] = value_for_existing_column(col, data)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def generate_measurement(
    elapsed_real_seconds: float,
    simulated_start_time: datetime,
    sim_seconds_per_real_second: float,
    in_train_event: bool,
    train_progress: float,
    damage: float,
) -> tuple[dict, float]:
    simulated_elapsed_seconds = elapsed_real_seconds * sim_seconds_per_real_second
    simulated_time = simulated_start_time + timedelta(seconds=simulated_elapsed_seconds)

    simulated_hour = (
        simulated_time.hour
        + simulated_time.minute / 60
        + simulated_time.second / 3600
    )

    # Temperatura diaria suave: mínima de madrugada, máxima hacia mediodía/tarde
    temperature = (
        18
        + 7 * math.sin(2 * math.pi * (simulated_hour - 6) / 24)
        + random.gauss(0, 0.25)
    )

    if in_train_event:
        # Envolvente para que el tren entre, tenga pico y salga
        envelope = math.sin(math.pi * train_progress)
        envelope = max(0.15, envelope)

        intensity = random.uniform(0.85, 1.25) * envelope

        accel_x = random.gauss(0, 0.03) + random.uniform(-0.35, 0.35) * intensity
        accel_y = random.gauss(0, 0.03) + random.uniform(-0.35, 0.35) * intensity
        accel_z = 1.0 + random.gauss(0, 0.02) + random.uniform(-0.55, 0.55) * intensity

        # Magnitud dinámica, quitando la gravedad aproximada de z
        accel_magnitude = math.sqrt(
            accel_x**2 + accel_y**2 + (accel_z - 1.0) ** 2
        )

        vibration = 0.15 + 1.25 * intensity + random.gauss(0, 0.05)
        piezo = 0.10 + 1.60 * intensity + random.gauss(0, 0.06)

        event_type = "train_passage"
        train_detected = 1

        # El daño crece más durante eventos de tren
        damage_increment = 0.0012 * intensity

    else:
        accel_x = random.gauss(0, 0.008)
        accel_y = random.gauss(0, 0.008)
        accel_z = 1.0 + random.gauss(0, 0.004)

        accel_magnitude = math.sqrt(
            accel_x**2 + accel_y**2 + (accel_z - 1.0) ** 2
        )

        vibration = max(0, random.gauss(0.025, 0.008))
        piezo = max(0, random.gauss(0.018, 0.007))

        event_type = "rest"
        train_detected = 0

        # Daño residual muy pequeño por paso del tiempo
        damage_increment = 0.00001

    damage = min(0.95, damage + damage_increment)
    life_percent = max(0, 100 * (1 - damage))

    data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "simulated_time": simulated_time.isoformat(timespec="seconds"),
        "simulated_hour": round(simulated_hour, 3),
        "source": DEMO_SOURCE,
        "sensor_id": SENSOR_ID,
        "event_type": event_type,
        "train_detected": train_detected,
        "temperature_C": round(temperature, 3),
        "accel_x": round(accel_x, 6),
        "accel_y": round(accel_y, 6),
        "accel_z": round(accel_z, 6),
        "accel_magnitude": round(accel_magnitude, 6),
        "vibration": round(vibration, 6),
        "piezo": round(piezo, 6),
        "damage": round(damage, 6),
        "life_percent": round(life_percent, 3),
    }

    return data, damage


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    fieldnames = ensure_csv_schema(CSV_PATH)

    print("============================================")
    print(" SIMULADOR DE DEMO INICIADO")
    print("============================================")
    print(f"CSV objetivo: {CSV_PATH}")
    print(f"Duración real: {DEMO_DURATION_SECONDS} s")
    print(f"Tiempo simulado: {SIMULATED_HOURS} h")
    print(f"Escritura cada: {UPDATE_EVERY_SECONDS} s")
    print("Pulsa CTRL + C para parar.")
    print("============================================")

    simulated_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    sim_seconds_per_real_second = (SIMULATED_HOURS * 3600) / DEMO_DURATION_SECONDS

    start_real = time.time()
    next_train_at = random.uniform(3, 6)
    train_active_until = -1
    train_started_at = -1

    damage = INITIAL_DAMAGE
    sample_count = 0
    train_count = 0

    try:
        while True:
            now = time.time()
            elapsed = now - start_real

            if elapsed > DEMO_DURATION_SECONDS:
                break

            if elapsed >= next_train_at and elapsed > train_active_until:
                train_started_at = elapsed
                train_active_until = elapsed + TRAIN_DURATION_SECONDS
                next_train_at = elapsed + random.uniform(*TRAIN_INTERVAL_RANGE_SECONDS)
                train_count += 1
                print(f"\n[TREN] Paso de tren #{train_count}")

            in_train_event = elapsed <= train_active_until

            if in_train_event:
                train_progress = (elapsed - train_started_at) / TRAIN_DURATION_SECONDS
                train_progress = min(max(train_progress, 0), 1)
            else:
                train_progress = 0

            data, damage = generate_measurement(
                elapsed_real_seconds=elapsed,
                simulated_start_time=simulated_start_time,
                sim_seconds_per_real_second=sim_seconds_per_real_second,
                in_train_event=in_train_event,
                train_progress=train_progress,
                damage=damage,
            )

            append_row(CSV_PATH, fieldnames, data)
            sample_count += 1

            status = "TREN" if data["train_detected"] == 1 else "reposo"

            print(
                f"\rFila {sample_count:03d} | "
                f"Hora sim: {data['simulated_time'][11:19]} | "
                f"Estado: {status:6s} | "
                f"T={data['temperature_C']:5.2f} ºC | "
                f"Vib={data['vibration']:.3f} | "
                f"Piezo={data['piezo']:.3f} | "
                f"Vida={data['life_percent']:6.2f} %",
                end="",
                flush=True,
            )

            time.sleep(UPDATE_EVERY_SECONDS)

    except KeyboardInterrupt:
        print("\n\n[STOP] Simulación detenida manualmente.")

    print("\n============================================")
    print(" DEMO FINALIZADA")
    print(f" Filas generadas: {sample_count}")
    print(f" Trenes simulados: {train_count}")
    print(f" Vida final: {100 * (1 - damage):.2f} %")
    print("============================================")


if __name__ == "__main__":
    main()