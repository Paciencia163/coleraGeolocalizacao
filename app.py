import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap, Draw
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------
# COMPATIBILIDADE COM STREAMLIT NOVO/ANTIGO
# -------------------------------------------------------
if not hasattr(st, "rerun"):
    st.rerun = st.experimental_rerun

# -------------------------------------------------------
# CONFIGURAÇÃO
# -------------------------------------------------------
st.set_page_config("🦠 Painel de Cólera — Angola", layout="wide")

# -------------------------------------------------------
# AUTENTICAÇÃO BÁSICA
# -------------------------------------------------------
USERS = {"admin": "1234", "gestor": "colera2025"}

def login():
    st.title("🔐 Acesso ao Painel de Monitoramento")
    user = st.text_input("Usuário")
    pwd = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if user in USERS and USERS[user] == pwd:
            st.session_state["auth"] = True
            st.session_state["user"] = user
            st.success("Login realizado com sucesso ✅")
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos.")
    st.stop()

if "auth" not in st.session_state or not st.session_state["auth"]:
    login()

st.sidebar.success(f"👤 Usuário: {st.session_state['user']}")
if st.sidebar.button("Sair"):
    st.session_state.clear()
    st.rerun()

# -------------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------------
def now(): 
    return datetime.utcnow()

def load_default_data():
    data = [
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

def load_csv(file):
    df = pd.read_csv(file)
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("CSV deve conter colunas 'lat' e 'lon'.")
    if "provincia" not in df.columns:
        df["provincia"] = "Desconhecida"
    if "cases" not in df.columns:
        df["cases"] = 1
    if "date" not in df.columns:
        df["date"] = now()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(now())
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(1)
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
    pad_lat, pad_lon = (lat_max - lat_min) * 0.2 or 0.01, (lon_max - lon_min) * 0.2 or 0.01

    lats = np.linspace(lat_min - pad_lat, lat_max + pad_lat, grid_size)
    lons = np.linspace(lon_min - pad_lon, lon_max + pad_lon, grid_size)
    mesh = np.meshgrid(lats, lons)
    pts = np.vstack([mesh[0].ravel(), mesh[1].ravel()]).T
    dens = np.exp(kde.score_samples(pts)).reshape(grid_size, grid_size)
    dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)
    return {"lats": lats, "lons": lons, "density": dens}

def add_heatmap(map_obj, kde_result, radius=15):
    if kde_result is None: return
    pts = []
    for i, lat in enumerate(kde_result["lats"]):
        for j, lon in enumerate(kde_result["lons"]):
            val = kde_result["density"][j, i]
            if val > 0.001: 
                pts.append([lat, lon, float(val)])
    HeatMap(pts, radius=radius, blur=20).add_to(map_obj)

# -------------------------------------------------------
# DADOS INICIAIS
# -------------------------------------------------------
if "cases_df" not in st.session_state:
    st.session_state["cases_df"] = load_default_data()
df = st.session_state["cases_df"]

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.header("⚙️ Configurações")
uploaded = st.sidebar.file_uploader("📂 Atualizar CSV", type=["csv"])
if uploaded:
    try:
        new_df = load_csv(uploaded)
        st.session_state["cases_df"] = pd.concat([df, new_df], ignore_index=True)
        st.sidebar.success(f"{len(new_df)} novos registros adicionados!")
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")

# -------------------------------------------------------
# ABAS
# -------------------------------------------------------
tab1, tab2 = st.tabs(["📍 Painel Principal", "📈 Análises e Previsão"])

# -------------------------------------------------------
# 🧭 TAB 1 - PAINEL PRINCIPAL
# -------------------------------------------------------
with tab1:
    st.subheader("🗺️ Mapa Interativo de Casos e Risco")

    col1, col2, col3 = st.columns(3)
    total = int(df["cases"].sum())
    media = round(df["cases"].mean(), 1)
    top = df.groupby("provincia")["cases"].sum().idxmax()
    col1.metric("Total de Casos", total)
    col2.metric("Média por Local", media)
    col3.metric("Província com mais casos", top)

    bandwidth = st.sidebar.slider("Bandwidth (KDE)", 0.005, 0.1, 0.02)
    half_life = st.sidebar.slider("Meia-vida temporal (dias)", 1, 60, 14)
    heat_radius = st.sidebar.slider("Raio do Heatmap", 5, 40, 15)

    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=6)
    Draw(export=True).add_to(m)

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=4 + np.log1p(r["cases"]),
            popup=f"{r['provincia']}<br>Casos: {r['cases']}<br>{r['date'].date()}",
            color="red",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    kde_now = compute_kde(df, bandwidth, 200, half_life)
    add_heatmap(m, kde_now, radius=heat_radius)
    st_folium(m, height=600)

    st.markdown("### 📝 Reportar Novo Caso")
    with st.form("novo_caso"):
        provincia = st.text_input("Província", "Nova Província")
        lat = st.number_input("Latitude", value=-8.8, format="%.6f")
        lon = st.number_input("Longitude", value=13.2, format="%.6f")
        cases = st.number_input("Número de casos", min_value=1, value=1)
        date = st.date_input("Data", datetime.now())
        submitted = st.form_submit_button("Registrar Caso")
        if submitted:
            new_row = {"provincia": provincia, "lat": lat, "lon": lon, "date": pd.to_datetime(date), "cases": cases}
            st.session_state["cases_df"] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"Caso registrado com sucesso em {provincia}!")

# -------------------------------------------------------
# 📈 TAB 2 - ANÁLISES E PREVISÃO
# -------------------------------------------------------
with tab2:
    st.subheader("📊 Análises e Tendências")

    df = st.session_state["cases_df"].copy()
    df_time = df.groupby("date")["cases"].sum().reset_index()
    df_time = df_time.sort_values("date")

    col1, col2 = st.columns(2)
    fig1 = px.line(df_time, x="date", y="cases", title="Evolução dos Casos", markers=True)
    col1.plotly_chart(fig1, use_container_width=True)

    df_prov = df.groupby("provincia")["cases"].sum().reset_index().sort_values("cases", ascending=False)
    fig2 = px.bar(df_prov, x="provincia", y="cases", color="cases", title="Casos por Província")
    col2.plotly_chart(fig2, use_container_width=True)

    # ---- Previsão ----
    st.markdown("### 🔮 Previsão de Casos")
    dias_prev = st.slider("Dias para previsão", 3, 30, 7)
    if len(df_time) > 3:
        X = np.arange(len(df_time)).reshape(-1, 1)
        y = np.log1p(df_time["cases"].values)
        model = LinearRegression().fit(X, y)

        future_X = np.arange(len(df_time) + dias_prev).reshape(-1, 1)
        preds = np.expm1(model.predict(future_X))
        dates = [df_time["date"].min() + timedelta(days=i) for i in range(len(future_X))]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_time["date"], y=df_time["cases"], mode="lines+markers", name="Casos reais"))
        fig3.add_trace(go.Scatter(x=dates, y=preds, mode="lines", name="Previsão", line=dict(dash="dash", color="orange")))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Poucos dados para gerar previsão.")

# -------------------------------------------------------
# EXPORTAÇÃO
# -------------------------------------------------------
st.markdown("---")
st.download_button("⬇️ Baixar CSV Consolidado", df.to_csv(index=False).encode("utf-8"), "casos_colera.csv", "text/csv")

st.caption("© 2025 — Painel de Monitoramento de Cólera (demo com dados simulados).")
