"""
=============================================================================
  ANÁLISE BLUESKY + MERCADO DE CAFÉ (KC=F)
  Autor: gerado automaticamente
  Descrição: Unifica CSVs do Bluesky, filtra colunas relevantes,
             faz análise exploratória completa e compara com o
             preço futuro do café Arábica (KC=F) via yfinance.
  Saída: C:/Users/joao3/Desktop/IC_Cafe/dados_graficos_bluesky/
=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import os
import ast
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import yfinance as yf
from pathlib import Path
from collections import Counter
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ── Configurações de caminho ───────────────────────────────────────────────
INPUT_DIR  = Path(r"C:\Users\joao3\Desktop\IC_Cafe\bluesky\BlueSky.py\data_with_score")
OUTPUT_DIR = Path(r"C:\Users\joao3\Desktop\IC_Cafe\dados_graficos_bluesky")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colunas que vamos MANTER após o filtro
COLUNAS_UTEIS = [
    "created_at",
    "text",
    "author_handle",
    "like_count",
    "repost_count",
    "reply_count",
    "financial_score",
    "top_topics",
    "top_scores",
]

# Paleta visual consistente
PALETA = {
    "primario":   "#2D6A4F",   # verde café
    "secundario": "#B5883B",   # dourado
    "acento":     "#C0392B",   # vermelho alerta
    "fundo":      "#F9F6F0",   # creme
    "texto":      "#1A1A1A",
    "neutro":     "#95A5A6",
}
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": PALETA["fundo"],
    "axes.facecolor":   PALETA["fundo"],
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

# ══════════════════════════════════════════════════════════════════════════
# 1. UNIFICAÇÃO DOS CSVs
# ══════════════════════════════════════════════════════════════════════════
def carregar_e_unificar(input_dir: Path) -> pd.DataFrame:
    arquivos = sorted(input_dir.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {input_dir}")

    print(f"📂 {len(arquivos)} arquivo(s) encontrado(s):")
    frames = []
    for arq in arquivos:
        try:
            df_tmp = pd.read_csv(arq, sep="\t", parse_dates=False, low_memory=False)
            # tenta tab; se só uma coluna, tenta vírgula
            if df_tmp.shape[1] == 1:
                df_tmp = pd.read_csv(arq, sep=",", parse_dates=False, low_memory=False)
            frames.append(df_tmp)
            print(f"   ✔ {arq.name}  ({len(df_tmp)} linhas)")
        except Exception as e:
            print(f"   ✘ {arq.name} — erro: {e}")

    df = pd.concat(frames, ignore_index=True)
    print(f"\n📊 Total bruto: {len(df)} linhas | {df.shape[1]} colunas")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. FILTRO E LIMPEZA
# ══════════════════════════════════════════════════════════════════════════
def filtrar_e_limpar(df: pd.DataFrame) -> pd.DataFrame:
    # Mantém só colunas que existem no dataframe
    colunas_presentes = [c for c in COLUNAS_UTEIS if c in df.columns]
    df = df[colunas_presentes].copy()

    # ── Datas ─────────────────────────────────────────────────────────────
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)

    # Coluna de data sem hora (para merge diário)
    df["date"] = df["created_at"].dt.normalize().dt.tz_localize(None)

    # ── Numéricos ─────────────────────────────────────────────────────────
    for col in ["like_count", "repost_count", "reply_count", "financial_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── top_topics: converte string → lista ──────────────────────────────
    if "top_topics" in df.columns:
        def parse_lista(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except Exception:
                    return []
            return []
        df["top_topics"] = df["top_topics"].apply(parse_lista)

    # ── top_scores: converte string → lista de floats ────────────────────
    if "top_scores" in df.columns:
        def parse_scores(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except Exception:
                    return []
            return []
        df["top_scores"] = df["top_scores"].apply(parse_scores)

    # ── Remove duplicatas exatas ─────────────────────────────────────────
    antes = len(df)
    df = df.drop_duplicates(subset=["created_at", "text"], keep="first")
    print(f"🧹 Duplicatas removidas: {antes - len(df)}")
    print(f"✅ Dataset limpo: {len(df)} linhas  |  período: "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 3. AGRUPAMENTO DIÁRIO (sentimento)
# ══════════════════════════════════════════════════════════════════════════
def agregar_diario(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("date").agg(
        posts          = ("financial_score", "count"),
        score_medio    = ("financial_score", "mean"),
        score_mediano  = ("financial_score", "median"),
        score_std      = ("financial_score", "std"),
        likes          = ("like_count",      "sum"),
        reposts        = ("repost_count",    "sum"),
        replies        = ("reply_count",     "sum"),
        engajamento    = ("like_count",      lambda x: x.sum() + df.loc[x.index, "repost_count"].sum()),
    ).reset_index()
    agg["score_std"] = agg["score_std"].fillna(0)
    return agg


# ══════════════════════════════════════════════════════════════════════════
# 4. DADOS DO CAFÉ (yfinance)
# ══════════════════════════════════════════════════════════════════════════
def baixar_cafe(data_inicio: str, data_fim: str) -> pd.DataFrame:
    print(f"\n☕ Baixando KC=F  {data_inicio} → {data_fim} ...")
    ticker = yf.Ticker("KC=F")
    cafe = ticker.history(start=data_inicio, end=data_fim, interval="1d")

    if cafe.empty:
        raise ValueError("Não foi possível baixar dados do KC=F. Verifique sua conexão.")

    cafe = cafe.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    cafe.columns = ["date", "open", "high", "low", "close", "volume"]
    cafe["date"] = pd.to_datetime(cafe["date"]).dt.tz_localize(None).dt.normalize()
    cafe["retorno_diario"] = cafe["close"].pct_change() * 100
    cafe["volatilidade_7d"] = cafe["close"].rolling(7).std()
    cafe["mm7"]  = cafe["close"].rolling(7).mean()
    cafe["mm21"] = cafe["close"].rolling(21).mean()
    print(f"   ✔ {len(cafe)} pregões carregados")
    return cafe


# ══════════════════════════════════════════════════════════════════════════
# 5. MERGE (sentimento × preço)
# ══════════════════════════════════════════════════════════════════════════
def merge_dados(diario: pd.DataFrame, cafe: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(diario, cafe, on="date", how="inner")
    print(f"\n🔗 Merge realizado: {len(merged)} dias em comum")
    if len(merged) < 5:
        print("   ⚠️  Poucos dias em comum — verifique se os períodos se sobrepõem.")
    return merged


# ══════════════════════════════════════════════════════════════════════════
# 6. ANÁLISES & GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════

def salvar(fig: plt.Figure, nome: str):
    caminho = OUTPUT_DIR / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾 {nome}")


# ── 6.1 Série temporal: score × preço do café ────────────────────────────
def plot_serie_temporal(merged: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Sentimento Financeiro (Bluesky) × Preço do Café (KC=F)",
                 fontsize=15, fontweight="bold", color=PALETA["texto"])

    # Score médio diário
    ax1.plot(merged["date"], merged["score_medio"],
             color=PALETA["primario"], lw=2, label="Score Médio")
    ax1.fill_between(merged["date"],
                     merged["score_medio"] - merged["score_std"],
                     merged["score_medio"] + merged["score_std"],
                     alpha=0.2, color=PALETA["primario"])
    ax1.axhline(merged["score_medio"].mean(), color=PALETA["neutro"],
                ls="--", lw=1, label=f"Média geral ({merged['score_medio'].mean():.3f})")
    ax1.set_ylabel("Score Financeiro")
    ax1.legend(fontsize=9)

    # Preço do café
    ax2.plot(merged["date"], merged["close"],
             color=PALETA["secundario"], lw=2, label="KC=F Close")
    ax2.plot(merged["date"], merged["mm7"],
             color=PALETA["acento"], lw=1.2, ls="--", label="MM7")
    ax2.plot(merged["date"], merged["mm21"],
             color=PALETA["primario"], lw=1.2, ls=":", label="MM21")
    ax2.set_ylabel("Preço (USD/lb × 100)")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    salvar(fig, "01_serie_temporal_score_cafe.png")


# ── 6.2 Volume de posts e engajamento diário ─────────────────────────────
def plot_volume_posts(diario: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(diario["date"], diario["posts"],
           color=PALETA["primario"], alpha=0.75, label="Posts/dia")
    ax2 = ax.twinx()
    ax2.plot(diario["date"], diario["likes"] + diario["reposts"],
             color=PALETA["secundario"], lw=2, label="Likes + Reposts")
    ax.set_title("Volume de Posts e Engajamento Diário", fontweight="bold")
    ax.set_ylabel("Nº de Posts")
    ax2.set_ylabel("Likes + Reposts")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    fig.tight_layout()
    salvar(fig, "02_volume_posts_engajamento.png")


# ── 6.3 Distribuição do financial_score ──────────────────────────────────
def plot_distribuicao_score(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribuição do Financial Score", fontweight="bold")

    sns.histplot(df["financial_score"], bins=40, kde=True,
                 color=PALETA["primario"], ax=axes[0])
    axes[0].set_title("Histograma + KDE")
    axes[0].set_xlabel("Financial Score")

    # Boxplot por mês
    df_tmp = df.copy()
    df_tmp["mes"] = df_tmp["date"].dt.to_period("M").astype(str)
    sns.boxplot(data=df_tmp, x="mes", y="financial_score",
                palette="Set2", ax=axes[1])
    axes[1].set_title("Boxplot por Mês")
    axes[1].set_xlabel("Mês")
    axes[1].set_ylabel("Financial Score")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    salvar(fig, "03_distribuicao_score.png")


# ── 6.4 Top tópicos mais citados ─────────────────────────────────────────
def plot_top_topicos(df: pd.DataFrame):
    todos_topicos = []
    for lista in df["top_topics"]:
        todos_topicos.extend(lista)

    if not todos_topicos:
        print("   ⚠️  Sem dados de top_topics para plotar.")
        return

    contagem = Counter(todos_topicos).most_common(20)
    topicos, freq = zip(*contagem)

    fig, ax = plt.subplots(figsize=(10, 7))
    cores = [PALETA["primario"] if i % 2 == 0 else PALETA["secundario"]
             for i in range(len(topicos))]
    bars = ax.barh(topicos[::-1], freq[::-1], color=cores[::-1])
    ax.set_title("Top 20 Tópicos Mais Citados", fontweight="bold")
    ax.set_xlabel("Frequência")
    for bar, v in zip(bars, freq[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=8)
    fig.tight_layout()
    salvar(fig, "04_top_topicos.png")


# ── 6.5 Correlação score × preço do café ─────────────────────────────────
def plot_correlacao(merged: pd.DataFrame):
    if len(merged) < 5:
        print("   ⚠️  Dados insuficientes para correlação.")
        return

    r_p, p_p = pearsonr(merged["score_medio"], merged["close"])
    r_s, p_s = spearmanr(merged["score_medio"], merged["close"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Correlação: Sentimento × Preço do Café (KC=F)", fontweight="bold")

    # Scatter
    axes[0].scatter(merged["score_medio"], merged["close"],
                    alpha=0.7, color=PALETA["primario"], edgecolors="white", lw=0.5, s=60)
    m, b = np.polyfit(merged["score_medio"], merged["close"], 1)
    x_line = np.linspace(merged["score_medio"].min(), merged["score_medio"].max(), 100)
    axes[0].plot(x_line, m * x_line + b, color=PALETA["acento"], lw=2, label="Regressão linear")
    axes[0].set_title(f"Pearson r={r_p:.3f} (p={p_p:.3f})\nSpearman ρ={r_s:.3f} (p={p_s:.3f})")
    axes[0].set_xlabel("Score Médio Diário")
    axes[0].set_ylabel("Preço KC=F")
    axes[0].legend(fontsize=9)

    # Heatmap de correlação geral
    cols_corr = ["score_medio", "posts", "likes", "reposts", "replies",
                 "close", "retorno_diario", "volatilidade_7d"]
    cols_corr = [c for c in cols_corr if c in merged.columns]
    corr_mat  = merged[cols_corr].corr()
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=axes[1],
                annot_kws={"size": 8})
    axes[1].set_title("Heatmap de Correlação")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    salvar(fig, "05_correlacao_score_cafe.png")


# ── 6.6 Retorno diário do café × score (lag analysis) ────────────────────
def plot_lag_analysis(merged: pd.DataFrame):
    if len(merged) < 10:
        print("   ⚠️  Dados insuficientes para análise de lag.")
        return

    lags   = range(-5, 6)
    corrs  = []
    for lag in lags:
        shifted = merged["score_medio"].shift(lag)
        validos = merged[["close"]].join(shifted.rename("score_lag")).dropna()
        if len(validos) > 3:
            r, _ = pearsonr(validos["score_lag"], validos["close"])
        else:
            r = np.nan
        corrs.append(r)

    fig, ax = plt.subplots(figsize=(10, 5))
    cores_lag = [PALETA["primario"] if c >= 0 else PALETA["acento"] for c in corrs]
    ax.bar(lags, corrs, color=cores_lag, alpha=0.8, edgecolor="white")
    ax.axhline(0, color=PALETA["texto"], lw=1)
    ax.set_title("Análise de Defasagem (Lag): Score vs Preço do Café",
                 fontweight="bold")
    ax.set_xlabel("Lag (dias)  —  negativo: score antecede preço")
    ax.set_ylabel("Correlação de Pearson")
    ax.set_xticks(list(lags))
    ax.set_xticklabels([f"lag {l:+d}" for l in lags], rotation=30, ha="right")
    fig.tight_layout()
    salvar(fig, "06_lag_analysis.png")


# ── 6.7 Score médio por dia da semana ────────────────────────────────────
def plot_dia_semana(df: pd.DataFrame):
    df_tmp = df.copy()
    df_tmp["dia_semana"] = df_tmp["date"].dt.day_name()
    ordem = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    labels_pt = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Padrões por Dia da Semana", fontweight="bold")

    # Score
    media_dia = (df_tmp.groupby("dia_semana")["financial_score"]
                 .mean().reindex(ordem))
    axes[0].bar(labels_pt, media_dia.values, color=PALETA["primario"], alpha=0.8)
    axes[0].set_title("Score Médio por Dia")
    axes[0].set_ylabel("Financial Score Médio")

    # Posts
    posts_dia = (df_tmp.groupby("dia_semana")["text"]
                 .count().reindex(ordem))
    axes[1].bar(labels_pt, posts_dia.values, color=PALETA["secundario"], alpha=0.8)
    axes[1].set_title("Volume de Posts por Dia")
    axes[1].set_ylabel("Nº de Posts")

    fig.tight_layout()
    salvar(fig, "07_padroes_dia_semana.png")


# ── 6.8 Preço do café: OHLC simplificado ─────────────────────────────────
def plot_ohlc(cafe: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("KC=F — Preço do Café Arábica (Futuros CBOT)", fontweight="bold")

    ax1.fill_between(cafe["date"], cafe["low"], cafe["high"],
                     alpha=0.2, color=PALETA["secundario"], label="Range H/L")
    ax1.plot(cafe["date"], cafe["close"],
             color=PALETA["secundario"], lw=2, label="Close")
    ax1.plot(cafe["date"], cafe["mm7"],
             color=PALETA["primario"], lw=1.5, ls="--", label="MM7")
    ax1.plot(cafe["date"], cafe["mm21"],
             color=PALETA["acento"], lw=1.5, ls=":", label="MM21")
    ax1.set_ylabel("Preço (USD/lb × 100)")
    ax1.legend(fontsize=9)

    # Retorno diário
    cores_ret = [PALETA["primario"] if r >= 0 else PALETA["acento"]
                 for r in cafe["retorno_diario"].fillna(0)]
    ax2.bar(cafe["date"], cafe["retorno_diario"], color=cores_ret, alpha=0.8)
    ax2.axhline(0, color=PALETA["texto"], lw=0.8)
    ax2.set_ylabel("Retorno Diário (%)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    salvar(fig, "08_preco_cafe_ohlc.png")


# ── 6.9 Score médio vs retorno futuro do café (D+1) ──────────────────────
def plot_score_vs_retorno_futuro(merged: pd.DataFrame):
    df_tmp = merged.copy()
    df_tmp["retorno_D1"] = df_tmp["retorno_diario"].shift(-1)
    df_tmp = df_tmp.dropna(subset=["retorno_D1"])

    if len(df_tmp) < 5:
        print("   ⚠️  Dados insuficientes para análise D+1.")
        return

    r, p = pearsonr(df_tmp["score_medio"], df_tmp["retorno_D1"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_tmp["score_medio"], df_tmp["retorno_D1"],
               alpha=0.7, color=PALETA["primario"],
               edgecolors="white", lw=0.5, s=60)
    m, b = np.polyfit(df_tmp["score_medio"], df_tmp["retorno_D1"], 1)
    x_l = np.linspace(df_tmp["score_medio"].min(), df_tmp["score_medio"].max(), 100)
    ax.plot(x_l, m * x_l + b, color=PALETA["acento"], lw=2)
    ax.axhline(0, color=PALETA["neutro"], lw=1, ls="--")
    ax.axvline(df_tmp["score_medio"].mean(), color=PALETA["neutro"], lw=1, ls="--")
    ax.set_title(f"Score Hoje × Retorno do Café Amanhã (D+1)\nPearson r={r:.3f}  p={p:.3f}",
                 fontweight="bold")
    ax.set_xlabel("Score Médio Diário")
    ax.set_ylabel("Retorno KC=F no dia seguinte (%)")
    fig.tight_layout()
    salvar(fig, "09_score_vs_retorno_futuro_D1.png")


# ══════════════════════════════════════════════════════════════════════════
# 7. RELATÓRIO ESTATÍSTICO (CSV + TXT)
# ══════════════════════════════════════════════════════════════════════════
def gerar_relatorio(df: pd.DataFrame, diario: pd.DataFrame,
                    cafe: pd.DataFrame, merged: pd.DataFrame):

    # ── CSV unificado filtrado ────────────────────────────────────────────
    df_export = df.drop(columns=["top_topics", "top_scores"], errors="ignore")
    df_export.to_csv(OUTPUT_DIR / "dados_unificados_filtrados.csv", index=False)
    print("   💾 dados_unificados_filtrados.csv")

    # ── CSV diário com café ───────────────────────────────────────────────
    merged.to_csv(OUTPUT_DIR / "dados_diarios_merged.csv", index=False)
    print("   💾 dados_diarios_merged.csv")

    # ── Relatório textual ─────────────────────────────────────────────────
    linhas = []
    sep = "=" * 65

    linhas += [
        sep,
        "  RELATÓRIO DE ANÁLISE — BLUESKY + CAFÉ (KC=F)",
        sep,
        "",
        f"Período dos posts   : {df['date'].min().date()} → {df['date'].max().date()}",
        f"Total de posts      : {len(df):,}",
        f"Autores únicos      : {df['author_handle'].nunique():,}",
        f"Dias com dados      : {df['date'].nunique()}",
        "",
        "── FINANCIAL SCORE ──────────────────────────────────────",
        f"  Média             : {df['financial_score'].mean():.4f}",
        f"  Mediana           : {df['financial_score'].median():.4f}",
        f"  Desvio padrão     : {df['financial_score'].std():.4f}",
        f"  Mínimo            : {df['financial_score'].min():.4f}",
        f"  Máximo            : {df['financial_score'].max():.4f}",
        f"  % score > 0.5     : {(df['financial_score'] > 0.5).mean()*100:.1f}%",
        "",
        "── ENGAJAMENTO ──────────────────────────────────────────",
        f"  Total likes       : {df['like_count'].sum():,.0f}",
        f"  Total reposts     : {df['repost_count'].sum():,.0f}",
        f"  Total replies     : {df['reply_count'].sum():,.0f}",
        f"  Likes/post (med.) : {df['like_count'].mean():.2f}",
        "",
    ]

    if not merged.empty:
        r_p, p_p = pearsonr(merged["score_medio"], merged["close"])
        r_s, p_s = spearmanr(merged["score_medio"], merged["close"])
        linhas += [
            "── KC=F (CAFÉ ARÁBICA) ──────────────────────────────────",
            f"  Período           : {cafe['date'].min().date()} → {cafe['date'].max().date()}",
            f"  Close mínimo      : {cafe['close'].min():.2f}",
            f"  Close máximo      : {cafe['close'].max():.2f}",
            f"  Close médio       : {cafe['close'].mean():.2f}",
            f"  Retorno período   : {((cafe['close'].iloc[-1]/cafe['close'].iloc[0])-1)*100:.2f}%",
            f"  Volatilidade (σ)  : {cafe['close'].std():.2f}",
            "",
            "── CORRELAÇÃO (dias sobrepostos) ────────────────────────",
            f"  Dias em comum     : {len(merged)}",
            f"  Pearson r         : {r_p:.4f}  (p={p_p:.4f})",
            f"  Spearman ρ        : {r_s:.4f}  (p={p_s:.4f})",
        ]
        if p_p < 0.05:
            linhas.append("  → Correlação de Pearson estatisticamente significativa (p<0.05)")
        else:
            linhas.append("  → Correlação de Pearson NÃO significativa (p≥0.05)")
    else:
        linhas.append("  ⚠️  Sem sobreposição de datas entre Bluesky e KC=F.")

    linhas += ["", sep, "Arquivos salvos em:", str(OUTPUT_DIR), sep]

    relatorio_txt = "\n".join(linhas)
    with open(OUTPUT_DIR / "relatorio_estatistico.txt", "w", encoding="utf-8") as f:
        f.write(relatorio_txt)
    print("   💾 relatorio_estatistico.txt")
    print("\n" + relatorio_txt)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 65)
    print("  ANÁLISE BLUESKY + CAFÉ KC=F")
    print("=" * 65 + "\n")

    # 1. Carregar e unificar
    df_raw = carregar_e_unificar(INPUT_DIR)

    # 2. Filtrar e limpar
    df = filtrar_e_limpar(df_raw)

    # 3. Agregação diária
    diario = agregar_diario(df)

    # 4. Baixar dados do café (com margem de ±7 dias)
    data_inicio = (df["date"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    data_fim    = (df["date"].max() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    cafe = baixar_cafe(data_inicio, data_fim)

    # 5. Merge
    merged = merge_dados(diario, cafe)

    # 6. Gráficos
    print("\n📈 Gerando gráficos...")
    plot_serie_temporal(merged)
    plot_volume_posts(diario)
    plot_distribuicao_score(df)
    if "top_topics" in df.columns:
        plot_top_topicos(df)
    plot_correlacao(merged)
    plot_lag_analysis(merged)
    plot_dia_semana(df)
    plot_ohlc(cafe)
    plot_score_vs_retorno_futuro(merged)

    # 7. Relatório
    print("\n📝 Gerando relatórios...")
    gerar_relatorio(df, diario, cafe, merged)

    print(f"\n✅ Concluído! Todos os arquivos estão em:\n   {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
