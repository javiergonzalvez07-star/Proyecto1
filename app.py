import streamlit as st
import pandas as pd
import os

RUTA_CSV = "data/dataset_final_maestro.csv"

st.set_page_config(page_title="Dashboard SHM", layout="wide")

st.title("Dashboard individual del dispositivo")

if not os.path.exists(RUTA_CSV):
    st.warning("Todavía no existe el archivo procesado.")
    st.stop()

df = pd.read_csv(RUTA_CSV)

if df.empty:
    st.warning("El archivo está vacío.")
    st.stop()

st.subheader("Últimas mediciones")
st.dataframe(df.tail(20), use_container_width=True)

ultima = df.iloc[-1]

col1, col2, col3 = st.columns(3)

with col1:
    if "Temp (°C)" in df.columns:
        st.metric("Temperatura actual", f"{ultima['Temp (°C)']:.2f} °C")

with col2:
    if "Accel_mag" in df.columns:
        st.metric("Vibración actual", f"{ultima['Accel_mag']:.2f}")

with col3:
    if "Piezo_proxy" in df.columns:
        st.metric("Piezo actual", f"{ultima['Piezo_proxy']:.2f}")

st.subheader("Evolución temporal")

if "Temp (°C)" in df.columns:
    st.line_chart(df["Temp (°C)"])

if "Accel_mag" in df.columns:
    st.line_chart(df["Accel_mag"])

if "Piezo_proxy" in df.columns:
    st.line_chart(df["Piezo_proxy"])