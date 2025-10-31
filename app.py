import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap, Draw
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from datetime import datetime

# -------------------------------------------------------
# CONFIGURAÇÃO
# -------------------------------------------------------
st.set_page_config("Cholera Risk Mapper (Provincias)", layout="wide")
st.title("🦠 Mapa de Risco de Cólera — Provincias com Casos Confirmados")

st.write(
    """
    Este aplicativo exibe casos simulados de cólera por província e permite atualizar os dados
    com um arquivo CSV. O mapa mostra as áreas de risco e permite projeções simples.
    """
)

# -------------------------------------------------------
# DADOS INICIAIS (SIMULADOS)
# -------------------------------------------------------
def load_default_data():
    data = [
        # lat, lon aproximados de capitais provinciais de Angola
        ["Luanda", -8.838, 13.234, "2025-10-20", 12],
        ["Benguela", -12.576, 13.405, "2025-10-22", 7],
        ["Huambo", -12.776, 15.739, "2025-10-18", 5],
        ["Lubango", -14.917, 13.492, "2025-10-19", 9],
        ["Malanje", -9.540, 16.350, "2025-10-21", 4],
        ["Uíge", -7.608, 15.061, "2025-10-23", 3],
        ["N'Dalatando", -9.296, 14.911, "2025-10-25", 5],
        ["Cabinda", -5.552, 12.197, "2025-10-17", 6],
        ["Sumbe", -11.205, 13.843, "2025-10-27", 4],
        ["Menongue", -14.658, 17.687, "2025-10-24", 2],
    ]
    df = pd.DataFrame(data, columns=["provincia", "lat", "lon", "date", "cases"])
    df["date"] = pd.to_datetime(df["date"])
    return df

# -------------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------------
def now():
    return datetime.utcnow()

def load_csv(file):
    df = pd.read_csv(file)
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("CSV deve conter colunas 'lat' e 'lon'.")
    if "date" not in df.columns:
        df["date"] = now()
    if "cases" not in df.columns:
        df["cases"] = 1
    df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(now())
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(1)
    if "provincia" not in df.columns:
        df["provincia"] = "Desconhecida"
    return df[["provincia", "lat", "lon", "date", "cases"]]

def temporal_weight(dates, half_life_days=14):
    base = now()
    weights = []
    for d in dates:
        delta = (base - pd.to_datetime(d)).days
        w = 0.5 ** (delta / half_life_days) if delta >= 0 else 1.0
        weights.append(w)
    return np.array(weights)

def compute_kde(df, bandwidth=0.02, grid_size=200, half_life_days=14):
    coords = df[["lat", "lon"]].to_numpy()
    if len(coords) == 0:
        return None
    weights = df["cases"].to_numpy() * temporal_weight(df["date"], half_life_days)
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(coords, sample_weight=weights)

    lat_min, lon_min = df["lat"].min(), df["lon"].min()
    lat_max, lon_max = df["lat"].max(), df["lon"].max()
    pad_lat = (lat_max - lat_min) * 0.2 or 0.01
    pad_lon = (lon_max - lon_min) * 0.2 or 0.01

    lats = np.linspace(lat_min - pad_lat, lat_max + pad_lat, grid_size)
    lons = np.linspace(lon_min - pad_lon, lon_max + pad_lon, grid_size)
    mesh = np.meshgrid(lats, lons)
    pts = np.vstack([mesh[0].ravel(), mesh[1].ravel()]).T

    log_dens = kde.score_samples(pts)
    dens = np.exp(log_dens).reshape(grid_size, grid_size)
    dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)
    return {"lats": lats, "lons": lons, "density": dens}

def add_heatmap(map_obj, kde_result, radius=15):
    if kde_result is None:
        return
    pts = []
    for i, lat in enumerate(kde_result["lats"]):
        for j, lon in enumerate(kde_result["lons"]):
            val = kde_result["density"][j, i]
            if val > 0.001:
                pts.append([lat, lon, float(val)])
    HeatMap(pts, radius=radius, blur=20).add_to(map_obj)

# -------------------------------------------------------
# INTERFACE
# -------------------------------------------------------
st.sidebar.header("⚙️ Parâmetros e Atualização")

uploaded = st.sidebar.file_uploader("Atualizar com novo CSV (lat, lon, date, cases)", type=["csv"])
bandwidth = st.sidebar.slider("Bandwidth (KDE)", 0.005, 0.1, 0.02)
half_life = st.sidebar.slider("Meia-vida temporal (dias)", 1, 60, 14)
heat_radius = st.sidebar.slider("Raio do Heatmap", 5, 40, 15)
predict_days = st.sidebar.slider("Projeção (dias à frente)", 1, 30, 7)
growth_rate = st.sidebar.number_input("Taxa de crescimento (%)", 0.0, 100.0, 5.0, step=1.0)

# -------------------------------------------------------
# ESTADO INICIAL
# -------------------------------------------------------
if "cases_df" not in st.session_state:
    st.session_state["cases_df"] = load_default_data()

# Atualizar com CSV (incremental)
if uploaded is not None:
    try:
        new_df = load_csv(uploaded)
        st.session_state["cases_df"] = pd.concat([st.session_state["cases_df"], new_df], ignore_index=True)
        st.success(f"Dados atualizados — {len(new_df)} novos registros adicionados.")
    except Exception as e:
        st.error(f"Erro ao carregar CSV: {e}")

# -------------------------------------------------------
# MAPA
# -------------------------------------------------------
st.subheader("🗺️ Mapa Interativo")

center = [
    st.session_state["cases_df"]["lat"].mean(),
    st.session_state["cases_df"]["lon"].mean(),
]
m = folium.Map(location=center, zoom_start=6)
Draw(export=True).add_to(m)

# adicionar pontos
for _, r in st.session_state["cases_df"].iterrows():
    folium.CircleMarker(
        location=[r["lat"], r["lon"]],
        radius=4 + np.log1p(r["cases"]),
        popup=f"{r['provincia']}<br>Casos: {r['cases']}<br>{r['date'].date()}",
        color="red",
        fill=True,
        fill_opacity=0.6,
    ).add_to(m)

# heatmap inicial
kde_now = compute_kde(st.session_state["cases_df"], bandwidth, 200, half_life)
add_heatmap(m, kde_now, radius=heat_radius)

map_data = st_folium(m, height=600, returned_objects=["last_clicked"])

# clique adiciona caso novo
if map_data.get("last_clicked"):
    lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    new_row = {"provincia": "Novo Caso", "lat": lat, "lon": lon, "date": now(), "cases": 1}
    st.session_state["cases_df"] = pd.concat(
        [st.session_state["cases_df"], pd.DataFrame([new_row])], ignore_index=True
    )
    st.success(f"Caso adicionado em ({lat:.4f}, {lon:.4f})")

# -------------------------------------------------------
# PROJEÇÃO
# -------------------------------------------------------
st.markdown("---")
st.subheader("📈 Projeção de Risco")

if st.button("Gerar projeção"):
    future_df = st.session_state["cases_df"].copy()
    future_df["cases"] *= (1 + growth_rate / 100.0)
    kde_future = compute_kde(future_df, bandwidth, 200, half_life)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Situação Atual**")
        m1 = folium.Map(location=center, zoom_start=6)
        add_heatmap(m1, kde_now, radius=heat_radius)
        st_folium(m1, height=400)
    with col2:
        st.write(f"**Projeção em {predict_days} dias (+{growth_rate:.1f}%)**")
        m2 = folium.Map(location=center, zoom_start=6)
        add_heatmap(m2, kde_future, radius=heat_radius)
        st_folium(m2, height=400)

# -------------------------------------------------------
# TABELA E EXPORTAÇÃO
# -------------------------------------------------------
st.markdown("---")
st.subheader("📋 Dados Consolidados")
if len(st.session_state["cases_df"]) > 0:
    st.dataframe(st.session_state["cases_df"].sort_values("date", ascending=False))
    csv = st.session_state["cases_df"].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV Consolidado", csv, "casos_colera_atualizados.csv", "text/csv")
else:
    st.info("Nenhum dado disponível.")

st.caption("© 2025 — Aplicativo demonstrativo com dados simulados por província.")
