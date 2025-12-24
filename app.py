import streamlit as st
import pandas as pd
import numpy as np
from ahp import ahp
from topsis import topsis

st.set_page_config(
    page_title="SPK Pemilihan Laptop",
    layout="wide"
)

st.title("💻 Sistem Pendukung Keputusan Pemilihan Laptop")
st.write("Metode AHP dan TOPSIS")

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("laptop.csv")

st.subheader("📊 Dataset Laptop (Asli)")
st.dataframe(df)

# ==========================
# PREPROCESS STORAGE
# ==========================
def convert_storage(value):
    value = value.upper()
    if "TB" in value:
        return int(value.replace("TB", "").strip()) * 1024
    elif "GB" in value:
        return int(value.replace("GB", "").strip())
    else:
        return 0

df["Storage"] = df["Storage"].apply(convert_storage)

st.subheader("📐 Dataset Setelah Preprocessing")
st.dataframe(df)

# ==========================
# KRITERIA
# ==========================
criteria = ["Price", "RAM", "Storage", "CPU_Speed", "Weight"]

# ==========================
# AHP
# ==========================
st.subheader("🔢 AHP - Penentuan Bobot Kriteria")

pairwise_matrix = np.array([
    [1,   3,   3,   4,   5],
    [1/3, 1,   2,   3,   4],
    [1/3, 1/2, 1,   2,   3],
    [1/4, 1/3, 1/2, 1,   2],
    [1/5, 1/4, 1/3, 1/2, 1]
])

weights, CI, CR = ahp(pairwise_matrix)

ahp_df = pd.DataFrame({
    "Kriteria": criteria,
    "Bobot": weights
})

st.table(ahp_df)
st.write(f"Consistency Index (CI): {CI:.4f}")
st.write(f"Consistency Ratio (CR): {CR:.4f}")

if CR < 0.1:
    st.success("Konsistensi AHP DITERIMA (CR < 0.1)")
else:
    st.error("Konsistensi AHP TIDAK DITERIMA")

# ==========================
# TOPSIS
# ==========================
st.subheader("🏆 TOPSIS - Hasil Perankingan Laptop")

data_topsis = df[criteria]

# Cost = -1, Benefit = 1
impacts = np.array([-1, 1, 1, 1, -1])

scores = topsis(data_topsis, weights, impacts)

df["Nilai Preferensi"] = scores
df_rank = df.sort_values("Nilai Preferensi", ascending=False)

st.dataframe(df_rank)

st.subheader("📈 Grafik Ranking Laptop")
st.bar_chart(
    df_rank.set_index("Brand")["Nilai Preferensi"]
)
