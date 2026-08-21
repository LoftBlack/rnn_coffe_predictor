"""
Open-Meteo - Extração e Análise de Dados Meteorológicos (Brasil)
================================================================
Sem token, sem cadastro — funciona direto!

Dependências:
    pip install requests pandas matplotlib seaborn scipy

Como usar:
    1. Configure LATITUDE, LONGITUDE, START_DATE, END_DATE
    2. Execute: python noaa_analise.py

Coordenadas úteis:
    Belo Horizonte : -19.9167, -43.9345
    São Paulo      : -23.5505, -46.6333
    Rio de Janeiro : -22.9068, -43.1729
    Brasília       : -15.7801, -47.9292
    Curitiba       : -25.4284, -49.2733
    Recife         : -8.0539,  -34.8811
"""

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────────
# CONFIGURAÇÕES — edite aqui
# ─────────────────────────────────────────────
LATITUDE   = -19.9167
LONGITUDE  = -43.9345
LOCAL_NOME = "Belo Horizonte, MG"
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

OUTPUT_DIR = "noaa_resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. EXTRAÇÃO
# ─────────────────────────────────────────────

def extrair() -> pd.DataFrame:
    print(f"\n{'='*52}")
    print(f" Local   : {LOCAL_NOME}")
    print(f" Período : {START_DATE}  →  {END_DATE}")
    print(f"{'='*52}")
    print("  → Buscando dados Open-Meteo...", end="", flush=True)

    url = "https://archive.open-meteo.com/v1/archive"
    params = {
        "latitude":        LATITUDE,
        "longitude":       LONGITUDE,
        "start_date":      START_DATE,
        "end_date":        END_DATE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max",
            "et0_fao_evapotranspiration",
        ],
        "timezone": "America/Sao_Paulo",
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["daily"]

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.columns = ["TMAX", "TMIN", "TMEAN", "PRCP", "VENTO", "ETP"]
    df.sort_index(inplace=True)

    n = df.dropna(how="all").shape[0]
    print(f" {n} dias carregados.")

    csv_path = os.path.join(OUTPUT_DIR, "dados_brutos.csv")
    df.to_csv(csv_path)
    print(f"  ✓ CSV salvo em: {csv_path}\n")
    return df


# ─────────────────────────────────────────────
# 2. ESTATÍSTICAS
# ─────────────────────────────────────────────

def estatisticas(df: pd.DataFrame):
    print(f"{'─'*52}")
    print(" ESTATÍSTICAS DESCRITIVAS")
    print(f"{'─'*52}")

    labels = {
        "TMAX":  "Temp. Máx (°C)",
        "TMIN":  "Temp. Mín (°C)",
        "TMEAN": "Temp. Média (°C)",
        "PRCP":  "Precipitação (mm)",
        "VENTO": "Vento Máx (km/h)",
        "ETP":   "Evapotranspiração (mm)",
    }
    cols = [c for c in labels if c in df.columns]
    st = df[cols].describe().T.round(2)
    st.index = [labels[c] for c in st.index]
    st.columns = ["n", "média", "dp", "mín", "Q1", "mediana", "Q3", "máx"]
    print(st.to_string())

    mensal = df["TMEAN"].resample("ME").mean()
    print(f"\n  Mês mais quente : {mensal.idxmax().strftime('%b/%Y')}  ({mensal.max():.1f}°C)")
    print(f"  Mês mais frio   : {mensal.idxmin().strftime('%b/%Y')}  ({mensal.min():.1f}°C)")

    extremos = df[df["PRCP"] >= 50]["PRCP"]
    print(f"  Dias com ≥50mm  : {len(extremos)}")
    if not extremos.empty:
        print(f"  Maior chuva     : {extremos.max():.1f} mm  ({extremos.idxmax().strftime('%d/%m/%Y')})")
    print()


# ─────────────────────────────────────────────
# 3. GRÁFICOS
# ─────────────────────────────────────────────

COR = {
    "azul":     "#2E6DA4",
    "escuro":   "#1A3A5C",
    "laranja":  "#E07B39",
    "vermelho": "#C0392B",
    "verde":    "#27AE60",
    "cinza":    "#7F8C8D",
    "fundo":    "#F7F9FB",
}

MESES = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

def _salvar(fig, nome):
    path = os.path.join(OUTPUT_DIR, nome)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_temperatura(df):
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=COR["fundo"])
    ax.set_facecolor(COR["fundo"])
    ax.fill_between(df.index, df["TMIN"], df["TMAX"],
                    color=COR["azul"], alpha=0.15, label="Faixa Tmín–Tmáx")
    mm = df["TMEAN"].rolling(30, center=True).mean()
    ax.plot(df.index, df["TMEAN"], color=COR["azul"], lw=0.5, alpha=0.35)
    ax.plot(mm.index, mm, color=COR["escuro"], lw=2, label="Média móvel 30d")
    ax.set_title(f"Temperatura — {LOCAL_NOME}", fontsize=13,
                 fontweight="bold", color=COR["escuro"])
    ax.set_ylabel("°C")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(); ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#ccc")
    fig.tight_layout()
    _salvar(fig, "01_temperatura.png")


def plot_precipitacao(df):
    mensal = df["PRCP"].resample("ME").sum()
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=COR["fundo"])
    ax.set_facecolor(COR["fundo"])
    ax.bar(mensal.index, mensal.values, width=25,
           color=COR["azul"], edgecolor="white", linewidth=0.3)
    ax.set_title(f"Precipitação Mensal Acumulada — {LOCAL_NOME}", fontsize=13,
                 fontweight="bold", color=COR["escuro"])
    ax.set_ylabel("mm")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#ccc")
    fig.tight_layout()
    _salvar(fig, "02_precipitacao.png")


def plot_climatologia(df):
    df2 = df.copy()
    df2["mes"] = df2.index.month
    mensal = df2.groupby("mes")[["TMAX","TMIN","TMEAN","PRCP"]].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), facecolor=COR["fundo"])
    for ax in (ax1, ax2):
        ax.set_facecolor(COR["fundo"])

    ax1.plot(mensal.index, mensal["TMAX"],  "o-", color=COR["vermelho"], lw=2, ms=5, label="Tmáx")
    ax1.plot(mensal.index, mensal["TMIN"],  "o-", color=COR["azul"],     lw=2, ms=5, label="Tmín")
    ax1.plot(mensal.index, mensal["TMEAN"], "o--",color=COR["cinza"],    lw=1.5, ms=4, label="Tméd")
    ax1.set_xticks(range(1,13)); ax1.set_xticklabels(MESES, fontsize=8)
    ax1.set_title("Temperatura média mensal", fontweight="bold", color=COR["escuro"])
    ax1.set_ylabel("°C"); ax1.legend()
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, color="#ccc")

    ax2.bar(mensal.index, mensal["PRCP"], color=COR["azul"], edgecolor="white")
    ax2.set_xticks(range(1,13)); ax2.set_xticklabels(MESES, fontsize=8)
    ax2.set_title("Precipitação média mensal", fontweight="bold", color=COR["escuro"])
    ax2.set_ylabel("mm")
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, color="#ccc")

    fig.suptitle(f"{LOCAL_NOME}  |  {START_DATE[:4]}–{END_DATE[:4]}",
                 fontsize=12, fontweight="bold", color=COR["escuro"])
    fig.tight_layout()
    _salvar(fig, "03_climatologia.png")


def plot_tendencia(df):
    anual = df["TMEAN"].resample("YE").mean().dropna()
    if len(anual) < 3:
        return
    x = anual.index.year.values
    y = anual.values
    slope, intercept, r, p, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=COR["fundo"])
    ax.set_facecolor(COR["fundo"])
    ax.bar(x, y, color=COR["azul"], alpha=0.7, label="Temp. média anual")
    ax.plot(x, intercept + slope * x, color=COR["vermelho"], lw=2,
            label=f"Tendência  {slope:+.3f}°C/ano  (p={p:.3f})")
    ax.set_title(f"Tendência Anual de Temperatura — {LOCAL_NOME}",
                 fontsize=13, fontweight="bold", color=COR["escuro"])
    ax.set_ylabel("°C"); ax.set_xlabel("Ano")
    ax.legend(); ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#ccc")
    fig.tight_layout()
    _salvar(fig, "04_tendencia.png")

    sig = "significativa (p<0.05)" if p < 0.05 else "não significativa (p≥0.05)"
    print(f"\n  Tendência: {slope:+.4f}°C/ano | R²={r**2:.3f} | {sig}")


def plot_heatmap(df):
    df2 = df["TMEAN"].copy().to_frame()
    df2["ano"] = df2.index.year
    df2["mes"] = df2.index.month
    pivot = df2.pivot_table(values="TMEAN", index="ano", columns="mes", aggfunc="mean")
    pivot.columns = MESES

    fig, ax = plt.subplots(figsize=(13, max(3, len(pivot) * 0.6 + 1)),
                           facecolor=COR["fundo"])
    sns.heatmap(pivot, ax=ax, cmap="RdYlBu_r", linewidths=0.4,
                annot=True, fmt=".1f", annot_kws={"size": 8},
                cbar_kws={"label": "°C"})
    ax.set_title(f"Temperatura Média Mensal (°C) — {LOCAL_NOME}",
                 fontsize=13, fontweight="bold", color=COR["escuro"])
    ax.set_ylabel("Ano"); ax.set_xlabel("")
    fig.tight_layout()
    _salvar(fig, "05_heatmap.png")


def plot_vento_etp(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), facecolor=COR["fundo"], sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor(COR["fundo"])

    mm_v = df["VENTO"].rolling(30, center=True).mean()
    ax1.plot(df.index, df["VENTO"], color=COR["cinza"], lw=0.5, alpha=0.4)
    ax1.plot(mm_v.index, mm_v, color=COR["laranja"], lw=2, label="Média móvel 30d")
    ax1.set_title(f"Velocidade do Vento Máx — {LOCAL_NOME}", fontweight="bold", color=COR["escuro"])
    ax1.set_ylabel("km/h"); ax1.legend()
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, color="#ccc")

    mm_e = df["ETP"].rolling(30, center=True).mean()
    ax2.plot(df.index, df["ETP"], color=COR["cinza"], lw=0.5, alpha=0.4)
    ax2.plot(mm_e.index, mm_e, color=COR["verde"], lw=2, label="Média móvel 30d")
    ax2.set_title("Evapotranspiração (ETP)", fontweight="bold", color=COR["escuro"])
    ax2.set_ylabel("mm/dia"); ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, color="#ccc")

    fig.tight_layout()
    _salvar(fig, "06_vento_etp.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    df = extrair()
    estatisticas(df)

    print("Gerando gráficos...")
    plot_temperatura(df)
    plot_precipitacao(df)
    plot_climatologia(df)
    plot_tendencia(df)
    plot_heatmap(df)
    plot_vento_etp(df)

    print(f"\n✅  Pronto! Resultados salvos em: ./{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()