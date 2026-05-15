"""
=============================================================================
  ANÁLISE BLUESKY + CAFÉ (KC=F) — DELAY +5 DIAS
  Hipótese: o sentimento financeiro do Bluesky de HOJE
            prevê o preço do café daqui 5 pregões.
  Saída: C:/Users/joao3/Desktop/IC_Cafe/dados_graficos_bluesky/Dados_delay/
=============================================================================
"""

import os
import ast
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ── Caminhos ───────────────────────────────────────────────────────────────
INPUT_DIR  = Path(r"C:\Users\joao3\Desktop\IC_Cafe\bluesky\BlueSky.py\data_with_score")
OUTPUT_DIR = Path(r"C:\Users\joao3\Desktop\IC_Cafe\dados_graficos_bluesky\Dados_delay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELAY = 5   # dias de antecedência testados

COLUNAS_UTEIS = [
    "created_at", "text", "author_handle",
    "like_count", "repost_count", "reply_count",
    "financial_score", "top_topics", "top_scores",
]

PALETA = {
    "primario":   "#2D6A4F",
    "secundario": "#B5883B",
    "acento":     "#C0392B",
    "info":       "#2471A3",
    "fundo":      "#F9F6F0",
    "texto":      "#1A1A1A",
    "neutro":     "#95A5A6",
    "positivo":   "#1E8449",
    "negativo":   "#C0392B",
}

plt.rcParams.update({
    "figure.facecolor": PALETA["fundo"],
    "axes.facecolor":   PALETA["fundo"],
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})
sns.set_theme(style="whitegrid", palette="muted")


# ══════════════════════════════════════════════════════════════════════════
# 1. CARGA E LIMPEZA (igual ao script anterior)
# ══════════════════════════════════════════════════════════════════════════
def carregar_e_unificar(input_dir: Path) -> pd.DataFrame:
    arquivos = sorted(input_dir.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {input_dir}")
    print(f"📂 {len(arquivos)} arquivo(s) encontrado(s):")
    frames = []
    for arq in arquivos:
        try:
            df_tmp = pd.read_csv(arq, sep="\t", low_memory=False)
            if df_tmp.shape[1] == 1:
                df_tmp = pd.read_csv(arq, sep=",", low_memory=False)
            frames.append(df_tmp)
            print(f"   ✔ {arq.name}  ({len(df_tmp)} linhas)")
        except Exception as e:
            print(f"   ✘ {arq.name} — erro: {e}")
    return pd.concat(frames, ignore_index=True)


def filtrar_e_limpar(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLUNAS_UTEIS if c in df.columns]
    df = df[cols].copy()

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    df["date"] = df["created_at"].dt.normalize().dt.tz_localize(None)

    for col in ["like_count", "repost_count", "reply_count", "financial_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    def parse_lista(val):
        if isinstance(val, list): return val
        try: return ast.literal_eval(val)
        except: return []

    if "top_topics" in df.columns:
        df["top_topics"] = df["top_topics"].apply(parse_lista)

    antes = len(df)
    df = df.drop_duplicates(subset=["created_at", "text"], keep="first")
    print(f"🧹 Duplicatas removidas: {antes - len(df)}")
    print(f"✅ Posts limpos: {len(df)}  |  {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def agregar_diario(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("date").agg(
        posts         = ("financial_score", "count"),
        score_medio   = ("financial_score", "mean"),
        score_mediano = ("financial_score", "median"),
        score_std     = ("financial_score", "std"),
        likes         = ("like_count",  "sum"),
        reposts       = ("repost_count","sum"),
        replies       = ("reply_count", "sum"),
    ).reset_index().fillna({"score_std": 0})


def baixar_cafe(data_inicio: str, data_fim: str) -> pd.DataFrame:
    print(f"\n☕ Baixando KC=F  {data_inicio} → {data_fim} ...")
    cafe = yf.Ticker("KC=F").history(start=data_inicio, end=data_fim, interval="1d")
    if cafe.empty:
        raise ValueError("Falha ao baixar KC=F. Verifique a conexão.")
    cafe = cafe.reset_index()[["Date","Open","High","Low","Close","Volume"]]
    cafe.columns = ["date","open","high","low","close","volume"]
    cafe["date"] = pd.to_datetime(cafe["date"]).dt.tz_localize(None).dt.normalize()
    cafe["retorno_diario"] = cafe["close"].pct_change() * 100
    cafe["mm7"]  = cafe["close"].rolling(7).mean()
    cafe["mm21"] = cafe["close"].rolling(21).mean()
    cafe["volatilidade_7d"] = cafe["close"].rolling(7).std()
    print(f"   ✔ {len(cafe)} pregões carregados")
    return cafe


# ══════════════════════════════════════════════════════════════════════════
# 2. CONSTRUÇÃO DO DATASET COM DELAY
#    score_hoje → preço/retorno daqui DELAY pregões
# ══════════════════════════════════════════════════════════════════════════
def build_delay_dataset(diario: pd.DataFrame, cafe: pd.DataFrame, delay: int) -> pd.DataFrame:
    """
    Para cada dia D com score, busca o preço do café no pregão D+delay.
    Usamos o calendário de pregões do KC=F para contar apenas dias úteis.
    """
    # Junta sentimento com preço no mesmo dia (D)
    base = pd.merge(diario, cafe[["date","close","retorno_diario","volatilidade_7d"]],
                    on="date", how="inner")

    # Cria coluna com preço futuro (D+delay pregões)
    cafe_futuro = cafe[["date","close","retorno_diario"]].copy()
    cafe_futuro.columns = ["date", f"close_D{delay}", f"retorno_D{delay}"]

    # Shifta o índice de pregões: para cada linha de 'base', procura a linha
    # correspondente em cafe_futuro deslocada 'delay' posições
    cafe_sorted = cafe.sort_values("date").reset_index(drop=True)
    cafe_sorted[f"date_D{delay}"] = cafe_sorted["date"].shift(-delay)

    lookup = cafe_sorted[["date", f"date_D{delay}"]].dropna()
    # adiciona preço futuro
    lookup = lookup.merge(
        cafe[["date","close","retorno_diario"]].rename(
            columns={"date": f"date_D{delay}",
                     "close": f"close_D{delay}",
                     "retorno_diario": f"retorno_D{delay}"}),
        on=f"date_D{delay}", how="left"
    )

    merged = base.merge(lookup, on="date", how="inner").dropna(
        subset=[f"close_D{delay}", f"retorno_D{delay}"])

    # Variação acumulada entre D e D+delay
    merged[f"var_acum_D{delay}"] = (
        (merged[f"close_D{delay}"] - merged["close"]) / merged["close"]
    ) * 100

    print(f"\n🔗 Dataset delay +{delay}d: {len(merged)} observações válidas")
    return merged


# ══════════════════════════════════════════════════════════════════════════
# 3. ESTATÍSTICAS PRINCIPAIS
# ══════════════════════════════════════════════════════════════════════════
def calcular_estatisticas(merged: pd.DataFrame, delay: int) -> dict:
    s = merged["score_medio"]
    r = merged[f"retorno_D{delay}"]
    v = merged[f"var_acum_D{delay}"]
    p = merged[f"close_D{delay}"]

    r_pear, p_pear   = pearsonr(s, r)
    r_spear, p_spear = spearmanr(s, r)
    r_pear_v, p_pear_v   = pearsonr(s, v)
    r_spear_v, p_spear_v = spearmanr(s, v)
    r_pear_p, p_pear_p   = pearsonr(s, p)

    # Regressão linear simples: score → retorno futuro
    X = s.values.reshape(-1, 1)
    reg_r = LinearRegression().fit(X, r.values)
    reg_v = LinearRegression().fit(X, v.values)

    # Teste: dias com score > mediana têm retorno futuro diferente?
    mediana = s.median()
    grupo_alto = r[s > mediana]
    grupo_baixo = r[s <= mediana]
    t_stat, t_p = ttest_ind(grupo_alto, grupo_baixo, equal_var=False)

    # Acurácia direcional: score alto (>mediana) → retorno positivo?
    acertos = ((s > mediana) & (r > 0)) | ((s <= mediana) & (r <= 0))
    acuracia_direcional = acertos.mean() * 100

    return {
        "n": len(merged),
        "delay": delay,
        # correlações com retorno D+delay
        "pearson_retorno": r_pear,     "p_pearson_retorno": p_pear,
        "spearman_retorno": r_spear,   "p_spearman_retorno": p_spear,
        # correlações com variação acumulada D→D+delay
        "pearson_var_acum": r_pear_v,  "p_pearson_var_acum": p_pear_v,
        "spearman_var_acum": r_spear_v,"p_spearman_var_acum": p_spear_v,
        # correlação com preço absoluto D+delay
        "pearson_preco": r_pear_p,     "p_pearson_preco": p_pear_p,
        # regressão
        "coef_reg_retorno": reg_r.coef_[0],   "r2_retorno": r2_score(r, reg_r.predict(X)),
        "coef_reg_var_acum": reg_v.coef_[0],  "r2_var_acum": r2_score(v, reg_v.predict(X)),
        # teste t
        "t_stat": t_stat, "t_p": t_p,
        "media_retorno_score_alto": grupo_alto.mean(),
        "media_retorno_score_baixo": grupo_baixo.mean(),
        # acurácia direcional
        "acuracia_direcional": acuracia_direcional,
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════
def salvar(fig, nome):
    fig.savefig(OUTPUT_DIR / nome, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾 {nome}")


# 4.1 Série temporal: score hoje × preço D+5
def plot_serie_temporal(merged: pd.DataFrame, delay: int):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                                         gridspec_kw={"height_ratios":[2,2,1]})
    fig.suptitle(f"Score Bluesky (hoje) × Preço do Café daqui {delay} pregões (D+{delay})",
                 fontsize=14, fontweight="bold")

    # Score
    ax1.plot(merged["date"], merged["score_medio"],
             color=PALETA["primario"], lw=2, label="Score médio")
    ax1.fill_between(merged["date"],
                     merged["score_medio"] - merged["score_std"],
                     merged["score_medio"] + merged["score_std"],
                     alpha=0.18, color=PALETA["primario"])
    ax1.axhline(merged["score_medio"].mean(), color=PALETA["neutro"],
                ls="--", lw=1, label=f"Média ({merged['score_medio'].mean():.3f})")
    ax1.set_ylabel("Score Financeiro (dia D)")
    ax1.legend(fontsize=9)
    ax1.set_title("Sentimento Bluesky — dia atual (D)", fontsize=11)

    # Preço futuro D+5
    ax2.plot(merged["date"], merged[f"close_D{delay}"],
             color=PALETA["secundario"], lw=2, label=f"Preço KC=F em D+{delay}")
    ax2.fill_between(merged["date"],
                     merged[f"close_D{delay}"].rolling(3).min(),
                     merged[f"close_D{delay}"].rolling(3).max(),
                     alpha=0.12, color=PALETA["secundario"])
    ax2.set_ylabel(f"Preço KC=F em D+{delay} (USD/lb×100)")
    ax2.legend(fontsize=9)
    ax2.set_title(f"Preço do Café {delay} pregões à frente (D+{delay})", fontsize=11)

    # Variação acumulada
    cores = [PALETA["positivo"] if v >= 0 else PALETA["negativo"]
             for v in merged[f"var_acum_D{delay}"]]
    ax3.bar(merged["date"], merged[f"var_acum_D{delay}"], color=cores, alpha=0.75)
    ax3.axhline(0, color=PALETA["texto"], lw=0.8)
    ax3.set_ylabel(f"Var. D→D+{delay} (%)")
    ax3.set_title(f"Variação acumulada D até D+{delay}", fontsize=11)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    salvar(fig, f"01_serie_temporal_delay{delay}.png")


# 4.2 Scatter: score hoje × retorno D+5
def plot_scatter_delay(merged: pd.DataFrame, delay: int, stats: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Score hoje × Retorno do Café em D+{delay}", fontweight="bold")

    for ax, (yvar, ylabel, rkey, pkey) in zip(axes, [
        (f"retorno_D{delay}", f"Retorno KC=F em D+{delay} (%)",
         "pearson_retorno", "p_pearson_retorno"),
        (f"var_acum_D{delay}", f"Variação acumulada D→D+{delay} (%)",
         "pearson_var_acum", "p_pearson_var_acum"),
    ]):
        ax.scatter(merged["score_medio"], merged[yvar],
                   alpha=0.6, color=PALETA["primario"],
                   edgecolors="white", lw=0.4, s=55)
        m, b = np.polyfit(merged["score_medio"], merged[yvar], 1)
        x_l = np.linspace(merged["score_medio"].min(), merged["score_medio"].max(), 100)
        ax.plot(x_l, m*x_l+b, color=PALETA["acento"], lw=2, label="Regressão linear")
        ax.axhline(0, color=PALETA["neutro"], lw=1, ls="--")
        ax.axvline(merged["score_medio"].mean(), color=PALETA["neutro"], lw=1, ls="--")
        r_val = stats[rkey]
        p_val = stats[pkey]
        sig = "✓ significativo" if p_val < 0.05 else "✗ não significativo"
        ax.set_title(f"Pearson r={r_val:.3f}  p={p_val:.3f}  {sig}", fontsize=10)
        ax.set_xlabel("Score médio (dia D)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    fig.tight_layout()
    salvar(fig, f"02_scatter_score_vs_retorno_D{delay}.png")


# 4.3 Score alto vs score baixo: retornos futuros
def plot_score_alto_baixo(merged: pd.DataFrame, delay: int, stats: dict):
    mediana = merged["score_medio"].median()
    alto  = merged[merged["score_medio"] >  mediana][f"retorno_D{delay}"]
    baixo = merged[merged["score_medio"] <= mediana][f"retorno_D{delay}"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Dias com Score Alto vs Baixo → Retorno em D+{delay}",
                 fontweight="bold")

    # Boxplot comparativo
    axes[0].boxplot([baixo, alto], labels=["Score baixo\n(≤mediana)", "Score alto\n(>mediana)"],
                    patch_artist=True,
                    boxprops=dict(facecolor=PALETA["fundo"], color=PALETA["texto"]),
                    medianprops=dict(color=PALETA["acento"], linewidth=2),
                    whiskerprops=dict(color=PALETA["neutro"]),
                    capprops=dict(color=PALETA["neutro"]),
                    flierprops=dict(marker="o", markerfacecolor=PALETA["neutro"],
                                    markersize=4, alpha=0.5))
    axes[0].axhline(0, color=PALETA["neutro"], lw=1, ls="--")
    axes[0].set_ylabel(f"Retorno KC=F em D+{delay} (%)")
    axes[0].set_title("Distribuição dos retornos futuros")

    # Médias com intervalo de confiança
    medias  = [baixo.mean(), alto.mean()]
    erros   = [baixo.sem() * 1.96, alto.sem() * 1.96]
    cores_b = [PALETA["neutro"], PALETA["primario"]]
    bars    = axes[1].bar(["Score baixo", "Score alto"], medias, color=cores_b,
                          alpha=0.8, edgecolor="white")
    axes[1].errorbar(["Score baixo", "Score alto"], medias, yerr=erros,
                     fmt="none", color=PALETA["texto"], capsize=6, lw=1.5)
    axes[1].axhline(0, color=PALETA["texto"], lw=0.8)
    for bar, v in zip(bars, medias):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + (0.05 if v >= 0 else -0.15),
                     f"{v:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    p_val = stats["t_p"]
    sig   = "✓ p<0.05" if p_val < 0.05 else f"✗ p={p_val:.3f}"
    axes[1].set_title(f"Retorno médio futuro (IC 95%)  —  teste-t: {sig}")
    axes[1].set_ylabel(f"Retorno médio em D+{delay} (%)")

    fig.tight_layout()
    salvar(fig, f"03_score_alto_vs_baixo_D{delay}.png")


# 4.4 Acurácia direcional por quintil de score
def plot_acuracia_quintil(merged: pd.DataFrame, delay: int):
    df = merged.copy()
    df["quintil"] = pd.qcut(df["score_medio"], 5,
                            labels=["Q1\n(mais baixo)","Q2","Q3","Q4","Q5\n(mais alto)"])
    resultado = df.groupby("quintil", observed=True).agg(
        acuracia = (f"retorno_D{delay}", lambda x: (x > 0).mean() * 100),
        n        = (f"retorno_D{delay}", "count"),
        retorno_medio = (f"retorno_D{delay}", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Acurácia Direcional e Retorno por Quintil de Score — D+{delay}",
                 fontweight="bold")

    cores_q = [PALETA["negativo"] if a < 50 else PALETA["positivo"]
               for a in resultado["acuracia"]]
    axes[0].bar(resultado["quintil"].astype(str), resultado["acuracia"],
                color=cores_q, alpha=0.82, edgecolor="white")
    axes[0].axhline(50, color=PALETA["neutro"], lw=1.5, ls="--",
                    label="50% (aleatório)")
    for i, (v, n) in enumerate(zip(resultado["acuracia"], resultado["n"])):
        axes[0].text(i, v + 1, f"{v:.1f}%\n(n={n})", ha="center", fontsize=9)
    axes[0].set_ylabel("Acurácia direcional (%)")
    axes[0].set_title("% de dias com retorno positivo em D+5")
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=9)

    cores_r = [PALETA["negativo"] if r < 0 else PALETA["positivo"]
               for r in resultado["retorno_medio"]]
    axes[1].bar(resultado["quintil"].astype(str), resultado["retorno_medio"],
                color=cores_r, alpha=0.82, edgecolor="white")
    axes[1].axhline(0, color=PALETA["texto"], lw=0.8)
    for i, v in enumerate(resultado["retorno_medio"]):
        axes[1].text(i, v + (0.03 if v >= 0 else -0.08),
                     f"{v:.2f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylabel(f"Retorno médio em D+{delay} (%)")
    axes[1].set_title("Retorno médio futuro por quintil de score")

    fig.tight_layout()
    salvar(fig, f"04_acuracia_quintil_D{delay}.png")


# 4.5 Heatmap de correlação completa
def plot_heatmap(merged: pd.DataFrame, delay: int):
    cols = ["score_medio", "score_std", "posts", "likes", "reposts",
            "close", f"close_D{delay}", f"retorno_D{delay}", f"var_acum_D{delay}",
            "volatilidade_7d"]
    cols = [c for c in cols if c in merged.columns]
    corr = merged[cols].corr()

    # Renomeia para legibilidade
    renomear = {
        "score_medio": "Score médio",
        "score_std": "Score std",
        "posts": "Nº posts",
        "likes": "Likes",
        "reposts": "Reposts",
        "close": "Preço D",
        f"close_D{delay}": f"Preço D+{delay}",
        f"retorno_D{delay}": f"Retorno D+{delay}",
        f"var_acum_D{delay}": f"Var. acum D+{delay}",
        "volatilidade_7d": "Volat. 7d",
    }
    corr = corr.rename(index=renomear, columns=renomear)

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.4, ax=ax, mask=mask, annot_kws={"size": 9},
                vmin=-1, vmax=1)
    ax.set_title(f"Heatmap de Correlação — todas as variáveis (delay +{delay}d)",
                 fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    salvar(fig, f"05_heatmap_correlacao_D{delay}.png")


# 4.6 Score vs variação acumulada com janela temporal
def plot_janela_temporal(merged: pd.DataFrame, delay: int):
    df = merged.copy()
    df["mes"] = df["date"].dt.to_period("M").astype(str)

    meses = sorted(df["mes"].unique())
    if len(meses) < 2:
        print("   ⚠️  Poucos meses para análise por janela.")
        return

    correlacoes_mes = []
    for mes in meses:
        sub = df[df["mes"] == mes]
        if len(sub) >= 5:
            r, p = pearsonr(sub["score_medio"], sub[f"retorno_D{delay}"])
            correlacoes_mes.append({"mes": mes, "pearson": r, "p": p, "n": len(sub)})

    if not correlacoes_mes:
        return

    res = pd.DataFrame(correlacoes_mes)
    cores = [PALETA["positivo"] if r >= 0 else PALETA["negativo"]
             for r in res["pearson"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(res["mes"], res["pearson"], color=cores, alpha=0.8, edgecolor="white")
    ax.axhline(0, color=PALETA["texto"], lw=0.8)
    ax.axhline(0.3,  color=PALETA["positivo"], lw=1, ls=":", alpha=0.6, label="r=+0.3")
    ax.axhline(-0.3, color=PALETA["negativo"], lw=1, ls=":", alpha=0.6, label="r=-0.3")
    for bar, (_, row) in zip(bars, res.iterrows()):
        sig = "*" if row["p"] < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.01 if bar.get_height() >= 0 else -0.04),
                f"{row['pearson']:.2f}{sig}", ha="center", fontsize=9)
    ax.set_title(f"Correlação de Pearson por Mês — Score D vs Retorno D+{delay}\n"
                 f"(* = p<0.05)", fontweight="bold")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Correlação de Pearson")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    salvar(fig, f"06_correlacao_por_mes_D{delay}.png")


# 4.7 Previsão vs realidade (regressão)
def plot_previsao_vs_real(merged: pd.DataFrame, delay: int, stats: dict):
    X = merged["score_medio"].values.reshape(-1, 1)
    y = merged[f"retorno_D{delay}"].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    residuos = y - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Regressão Linear: Score → Retorno D+{delay}", fontweight="bold")

    # Previsto vs Real
    axes[0].scatter(y, y_pred, alpha=0.6, color=PALETA["info"],
                    edgecolors="white", lw=0.4, s=55)
    lim = max(abs(y).max(), abs(y_pred).max()) * 1.1
    axes[0].plot([-lim, lim], [-lim, lim], color=PALETA["acento"],
                 lw=1.5, ls="--", label="Linha perfeita")
    r2 = stats["r2_retorno"]
    axes[0].set_title(f"Previsto vs Real  |  R²={r2:.4f}")
    axes[0].set_xlabel(f"Retorno real em D+{delay} (%)")
    axes[0].set_ylabel("Retorno previsto pelo score (%)")
    axes[0].legend(fontsize=9)

    # Resíduos
    axes[1].scatter(y_pred, residuos, alpha=0.6, color=PALETA["secundario"],
                    edgecolors="white", lw=0.4, s=55)
    axes[1].axhline(0, color=PALETA["acento"], lw=1.5, ls="--")
    axes[1].set_title("Resíduos da Regressão")
    axes[1].set_xlabel("Retorno previsto (%)")
    axes[1].set_ylabel("Resíduo (real − previsto) (%)")

    fig.tight_layout()
    salvar(fig, f"07_regressao_previsao_D{delay}.png")


# ══════════════════════════════════════════════════════════════════════════
# 5. RELATÓRIO
# ══════════════════════════════════════════════════════════════════════════
def gerar_relatorio(merged: pd.DataFrame, stats: dict, delay: int):
    merged.to_csv(OUTPUT_DIR / f"dados_delay{delay}_merged.csv", index=False)
    print(f"   💾 dados_delay{delay}_merged.csv")

    sep = "=" * 68
    def sig(p): return "✓ SIGNIFICATIVO (p<0.05)" if p < 0.05 else "✗ não significativo"

    linhas = [
        sep,
        f"  ANÁLISE COM DELAY +{delay} DIAS — Score Bluesky → Café KC=F",
        sep,
        "",
        f"Hipótese testada : score de HOJE prevê preço daqui {delay} pregões",
        f"Observações válidas: {stats['n']}",
        f"Período            : {merged['date'].min().date()} → {merged['date'].max().date()}",
        "",
        f"── CORRELAÇÃO: Score (D) × Retorno KC=F (D+{delay}) ─────────────────",
        f"  Pearson r  = {stats['pearson_retorno']:+.4f}   p={stats['p_pearson_retorno']:.4f}  {sig(stats['p_pearson_retorno'])}",
        f"  Spearman ρ = {stats['spearman_retorno']:+.4f}   p={stats['p_spearman_retorno']:.4f}  {sig(stats['p_spearman_retorno'])}",
        "",
        f"── CORRELAÇÃO: Score (D) × Var. Acumulada D→D+{delay} (%) ────────────",
        f"  Pearson r  = {stats['pearson_var_acum']:+.4f}   p={stats['p_pearson_var_acum']:.4f}  {sig(stats['p_pearson_var_acum'])}",
        f"  Spearman ρ = {stats['spearman_var_acum']:+.4f}   p={stats['p_spearman_var_acum']:.4f}  {sig(stats['p_spearman_var_acum'])}",
        "",
        f"── CORRELAÇÃO: Score (D) × Preço absoluto D+{delay} ─────────────────",
        f"  Pearson r  = {stats['pearson_preco']:+.4f}   p={stats['p_pearson_preco']:.4f}  {sig(stats['p_pearson_preco'])}",
        "",
        f"── REGRESSÃO LINEAR (Score → Retorno D+{delay}) ──────────────────────",
        f"  Coeficiente: {stats['coef_reg_retorno']:+.4f}  (cada +0.1 no score → {stats['coef_reg_retorno']*0.1:+.3f}% no retorno)",
        f"  R²         : {stats['r2_retorno']:.4f}  (o score explica {stats['r2_retorno']*100:.2f}% da variância do retorno)",
        "",
        f"── REGRESSÃO LINEAR (Score → Var. Acumulada D+{delay}) ───────────────",
        f"  Coeficiente: {stats['coef_reg_var_acum']:+.4f}",
        f"  R²         : {stats['r2_var_acum']:.4f}  ({stats['r2_var_acum']*100:.2f}% da variância explicada)",
        "",
        f"── TESTE-T: Score alto vs Score baixo ───────────────────────────────",
        f"  Retorno médio (score alto)  : {stats['media_retorno_score_alto']:+.4f}%",
        f"  Retorno médio (score baixo) : {stats['media_retorno_score_baixo']:+.4f}%",
        f"  Diferença                   : {stats['media_retorno_score_alto']-stats['media_retorno_score_baixo']:+.4f}%",
        f"  t-stat={stats['t_stat']:.3f}  p={stats['t_p']:.4f}  {sig(stats['t_p'])}",
        "",
        f"── ACURÁCIA DIRECIONAL ───────────────────────────────────────────────",
        f"  Score alto → retorno positivo (ou baixo → negativo): {stats['acuracia_direcional']:.1f}%",
        f"  (referência aleatória: 50%)",
        "",
        "── INTERPRETAÇÃO ────────────────────────────────────────────────────",
    ]

    r = stats["pearson_retorno"]
    p = stats["p_pearson_retorno"]
    ac = stats["acuracia_direcional"]

    if p < 0.05 and abs(r) >= 0.3:
        linhas.append(f"  ✅ Correlação moderada a forte E significativa: o sentimento")
        linhas.append(f"     do Bluesky aparenta ter poder preditivo sobre o KC=F em +{delay}d.")
    elif p < 0.05 and abs(r) < 0.3:
        linhas.append(f"  ⚠️  Correlação fraca mas estatisticamente significativa.")
        linhas.append(f"     Existe relação, mas com baixo poder explicativo individual.")
    else:
        linhas.append(f"  ❌ Correlação não significativa: não há evidência de que")
        linhas.append(f"     o score antecipa o retorno do café em +{delay} pregões.")

    if ac > 55:
        linhas.append(f"  📈 Acurácia direcional de {ac:.1f}% é acima do aleatório — relevante.")
    elif ac > 50:
        linhas.append(f"  📊 Acurácia direcional de {ac:.1f}% levemente acima do acaso.")
    else:
        linhas.append(f"  📉 Acurácia direcional abaixo de 50% — sem poder de direção.")

    linhas += ["", sep, f"Arquivos salvos em: {OUTPUT_DIR}", sep]

    txt = "\n".join(linhas)
    with open(OUTPUT_DIR / f"relatorio_delay{delay}.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"   💾 relatorio_delay{delay}.txt")
    print("\n" + txt)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*68)
    print(f"  ANÁLISE COM DELAY +{DELAY} DIAS — Bluesky → Café KC=F")
    print("="*68 + "\n")

    # Carga
    df_raw = carregar_e_unificar(INPUT_DIR)
    df     = filtrar_e_limpar(df_raw)
    diario = agregar_diario(df)

    # KC=F com margem extra para cobrir D+DELAY além do último dia de score
    data_inicio = (df["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    data_fim    = (df["date"].max() + pd.Timedelta(days=DELAY + 10)).strftime("%Y-%m-%d")
    cafe = baixar_cafe(data_inicio, data_fim)

    # Dataset com delay
    merged = build_delay_dataset(diario, cafe, DELAY)

    if len(merged) < 10:
        print("\n⚠️  Menos de 10 observações após o merge com delay.")
        print("   Verifique se os CSVs têm dados suficientemente recentes.")
        return

    # Estatísticas
    stats = calcular_estatisticas(merged, DELAY)

    # Gráficos
    print("\n📈 Gerando gráficos...")
    plot_serie_temporal(merged, DELAY)
    plot_scatter_delay(merged, DELAY, stats)
    plot_score_alto_baixo(merged, DELAY, stats)
    plot_acuracia_quintil(merged, DELAY)
    plot_heatmap(merged, DELAY)
    plot_janela_temporal(merged, DELAY)
    plot_previsao_vs_real(merged, DELAY, stats)

    # Relatório
    print("\n📝 Gerando relatório...")
    gerar_relatorio(merged, stats, DELAY)

    print(f"\n✅ Concluído! Arquivos em:\n   {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()