from machine import Pin, I2C, ADC
import time

# =========================================================
# CONFIGURACION GENERAL
# =========================================================

# MPU6050
I2C_SDA = 21
I2C_SCL = 22
MPU6050_ADDR = 0x68

# MAX6675
MAX6675_SCK = 18
MAX6675_SO = 19
MAX6675_CS = 5

# PIEZO ANALOGICO
PIEZO_PIN = 34

# TIEMPO ENTRE MEDIDAS
INTERVAL_MS = 500

# GUARDAR TAMBIEN EN ARCHIVO LOCAL
SAVE_TO_FILE = True
CSV_FILE = "datos_brutos.csv"


# =========================================================
# MPU6050
# =========================================================

class MPU6050:
    def __init__(self, i2c, addr=0x68):
        self.i2c = i2c
        self.addr = addr

        # Despertar sensor
        self.i2c.writeto_mem(self.addr, 0x6B, b'\x00')

        # Acelerometro ±2g
        self.i2c.writeto_mem(self.addr, 0x1C, b'\x00')

        # Giroscopio ±250 °/s
        self.i2c.writeto_mem(self.addr, 0x1B, b'\x00')

    def _read_word(self, reg):
        data = self.i2c.readfrom_mem(self.addr, reg, 2)
        value = (data[0] << 8) | data[1]
        if value & 0x8000:
            value -= 65536
        return value

    def read_accel(self):
        ax = self._read_word(0x3B) / 16384.0
        ay = self._read_word(0x3D) / 16384.0
        az = self._read_word(0x3F) / 16384.0
        return ax, ay, az

    def read_gyro(self):
        gx = self._read_word(0x43) / 131.0
        gy = self._read_word(0x45) / 131.0
        gz = self._read_word(0x47) / 131.0
        return gx, gy, gz


# =========================================================
# MAX6675
# =========================================================

class MAX6675:
    def __init__(self, sck_pin, so_pin, cs_pin):
        self.sck = Pin(sck_pin, Pin.OUT)
        self.so = Pin(so_pin, Pin.IN)
        self.cs = Pin(cs_pin, Pin.OUT)

        self.cs.value(1)
        self.sck.value(0)

    def read_temp_c(self):
        self.cs.value(0)
        time.sleep_us(10)

        value = 0
        for _ in range(16):
            self.sck.value(1)
            time.sleep_us(1)
            value = (value << 1) | self.so.value()
            self.sck.value(0)
            time.sleep_us(1)

        self.cs.value(1)

        # Si el termopar no está conectado
        if value & 0x4:
            return None

        value >>= 3
        return value * 0.25


# =========================================================
# CSV
# =========================================================

HEADER = "timestamp_ms,temp_c,ax,ay,az,gx,gy,gz,piezo_raw"

def init_csv_file():
    if not SAVE_TO_FILE:
        return

    try:
        with open(CSV_FILE, "r") as f:
            first_line = f.readline().strip()
            if first_line == HEADER:
                return
    except:
        pass

    with open(CSV_FILE, "w") as f:
        f.write(HEADER + "\n")

def append_csv_line(line):
    if not SAVE_TO_FILE:
        return
    with open(CSV_FILE, "a") as f:
        f.write(line + "\n")


# =========================================================
# SETUP
# =========================================================

i2c = I2C(0, scl=Pin(I2C_SCL), sda=Pin(I2C_SDA), freq=400000)
mpu = MPU6050(i2c)

max6675 = MAX6675(MAX6675_SCK, MAX6675_SO, MAX6675_CS)

piezo = ADC(Pin(PIEZO_PIN))
piezo.atten(ADC.ATTN_11DB)
piezo.width(ADC.WIDTH_12BIT)

init_csv_file()

# Imprimimos cabecera por serial una sola vez
print(HEADER)


# =========================================================
# LOOP PRINCIPAL
# =========================================================

while True:
    try:
        timestamp_ms = time.ticks_ms()

        temp_c = max6675.read_temp_c()
        ax, ay, az = mpu.read_accel()
        gx, gy, gz = mpu.read_gyro()
        piezo_raw = piezo.read()

        temp_str = "" if temp_c is None else "{:.2f}".format(temp_c)

        line = "{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{}".format(
            timestamp_ms,
            temp_str,
            ax, ay, az,
            gx, gy, gz,
            piezo_raw
        )

        # 1) Salida serial para demo y futuro dashboard
        print(line)

        # 2) Guardado opcional a archivo local
        append_csv_line(line)

    except Exception as e:
        print("ERROR,{}".format(str(e)))

    time.sleep_ms(INTERVAL_MS)