

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # permite salvar gráficos sem abrir janelas
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


# ============================================================
# CONFIGURAÇÕES
# ============================================================
BASE_DIR = Path(r"C:\Users\joao3\Desktop\IC_Cafe")
PATH_RAW_DATA = BASE_DIR / "noaa_resultados" / "dados_brutos.csv"
RESULTS_DIR = BASE_DIR / "processed_data"
PLOTS_DIR = RESULTS_DIR / "graficos_eda"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "noaa_analysis.log"

# Subpastas novas (organização das saídas)
DATA_DIR = RESULTS_DIR / "dados"
TABLES_DIR = RESULTS_DIR / "tabelas"
REPORT_DIR = RESULTS_DIR / "relatorios"

GLOBAL_START_DATE = "2020-01-01"
GLOBAL_END_DATE = "2025-12-31"

# Café arábica: futuro contínuo ICE (cotado em centavos de dólar por libra-peso).
# Obs.: contínuo = contrato mais próximo; pode haver pequenos saltos na troca de vencimento.
TICKER_CAFE = "KC=F"

# Outras séries que você já usou nos modelos (opcional, descomente para baixar):
# TICKERS_EXTRAS = {"milho_usc_bushel": "ZC=F", "petroleo_wti_usd_barril": "CL=F", "dolar_brl": "USDBRL=X"}

# Nome da coluna de data no CSV de clima (se não achar, tenta a primeira coluna).
DATE_CANDIDATES = ["time", "date", "data", "datetime", "timestamp", "datahora", "data_hora"]

# Dicionário das colunas do CSV original -> nome final, rótulo (gráficos), descrição e agregação mensal.
# ATENÇÃO: confira a unidade do VENTO (km/h é o padrão do Open-Meteo; m/s é comum em NOAA).
COLUNAS_CLIMA = {
    "TMAX":  dict(nome="temp_max_c", rotulo="T máxima (°C)",
                  descricao="Maior T do ar no dia", unidade="°C", mensal="mean"),
    "TMIN":  dict(nome="temp_min_c", rotulo="T mínima (°C)",
                  descricao="Menor T do ar no dia", unidade="°C", mensal="mean"),
    "TMEAN": dict(nome="temp_media_c", rotulo="T média (°C)",
                  descricao="T média do ar no dia", unidade="°C", mensal="mean"),
    "PRCP":  dict(nome="precipitacao_mm", rotulo="Precipitação (mm/dia)",
                  descricao="Chuva acumulada no dia", unidade="mm", mensal="sum"),
    "VENTO": dict(nome="vento_vel", rotulo="Vel do vento (km/h)",
                  descricao="Vel do vento (confira a unidade na fonte)", unidade="km/h", mensal="mean"),
    "ETP":   dict(nome="evapotranspiracao_mm", rotulo="Evapotranspiração (mm/dia)",
                  descricao="Evapotranspiração potencial de referência no dia", unidade="mm", mensal="sum"),
}

COLUNAS_CAFE = {
    "Open":   dict(nome="cafe_abertura_usc_lb", rotulo="Café – abertura ($US)",
                   descricao="Preço de abertura do futuro de café arábica", unidade="$US"),
    "High":   dict(nome="cafe_maxima_usc_lb", rotulo="Café – máxima ($US)",
                   descricao="Preço máximo do dia", unidade="$US"),
    "Low":    dict(nome="cafe_minima_usc_lb", rotulo="Café – mínima ($US)",
                   descricao="Preço mínimo do dia", unidade="$US"),
    "Close":  dict(nome="cafe_fechamento_usc_lb", rotulo="Café – fechamento ($US)",
                   descricao="Preço de fechamento (usado como preço de referência)", unidade="$US"),
    "Volume": dict(nome="cafe_volume_contratos", rotulo="Café – volume (contratos)",
                   descricao="Contratos negociados no dia", unidade="contratos"),
}

COL_PRECO = "cafe_fechamento_usc_lb"
COL_RETORNO = "cafe_retorno_diario_pct"
COL_PRECO_VAR_MENSAL = "cafe_variacao_mensal_pct"
MAX_DEFASAGEM_MESES = 12

NOMES_CLIMA = [v["nome"] for v in COLUNAS_CLIMA.values()]

ROTULOS = {v["nome"]: v["rotulo"] for v in COLUNAS_CLIMA.values()}
ROTULOS.update({v["nome"]: v["rotulo"] for v in COLUNAS_CAFE.values()})
ROTULOS[COL_RETORNO] = "Café – retorno diário (%)"
ROTULOS[COL_PRECO_VAR_MENSAL] = "Café – variação mensal (%)"


def rot(coluna):
    return ROTULOS.get(coluna, coluna)


# ============================================================
# PASTAS E LOGGING
# ============================================================
SUBPASTAS_GRAFICOS = {
    "clima": PLOTS_DIR / "clima",
    "cafe": PLOTS_DIR / "cafe",
    "distribuicoes": PLOTS_DIR / "distribuicoes",
    "correlacoes": PLOTS_DIR / "correlacoes",
    "clima_vs_cafe": PLOTS_DIR / "clima_vs_cafe",
}

for pasta in [LOG_DIR, RESULTS_DIR, DATA_DIR, TABLES_DIR, REPORT_DIR, PLOTS_DIR, *SUBPASTAS_GRAFICOS.values()]:
    pasta.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# FUNÇÕES AUXILIARES DE SAÍDA
# ============================================================
def padronizar_nome_coluna(nome):
    """Remove espaços nas pontas e troca espaços internos por '_'."""
    return re.sub(r"\s+", "_", str(nome).strip())


def salvar_tabela(df, nome, pasta=TABLES_DIR, indice=True):
    caminho = pasta / nome
    df.to_csv(caminho, index=indice, encoding="utf-8-sig")
    logger.info("Tabela salva: %s", caminho)


def salvar_fig(fig, grupo, nome):
    caminho = SUBPASTAS_GRAFICOS[grupo] / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico salvo: %s", caminho)


# ============================================================
# CARGA: CLIMA
# ============================================================
def localizar_coluna_data(df):
    mapa = {str(c).strip().lower(): c for c in df.columns}
    for candidato in DATE_CANDIDATES:
        if candidato in mapa:
            return mapa[candidato]
    # Última tentativa: primeira coluna parece conter datas?
    primeira = df.columns[0]
    if pd.to_datetime(df[primeira], errors="coerce").notna().mean() >= 0.7:
        return primeira
    return None


def converter_data(serie):
    convertida = pd.to_datetime(serie, errors="coerce")
    if convertida.notna().mean() < 0.7:  # talvez formato dd/mm/aaaa
        convertida = pd.to_datetime(serie, errors="coerce", dayfirst=True)
    return convertida


def carregar_clima(caminho):
    """Lê o CSV de clima, renomeia colunas para nomes descritivos e filtra o período."""
    logger.info("Lendo clima: %s", caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            "Confira se dados_brutos.csv está dentro da pasta noaa_resultados."
        )

    try:
        df = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(caminho, sep=None, engine="python", encoding="latin-1")

    if df.empty:
        raise ValueError("O CSV foi lido, mas não contém linhas.")

    df.columns = [padronizar_nome_coluna(c) for c in df.columns]
    logger.info("Dimensões iniciais: %s linhas x %s colunas", *df.shape)

    coluna_data = localizar_coluna_data(df)
    if coluna_data is None:
        raise ValueError("Não foi possível identificar a coluna de data no CSV de clima.")
    logger.info("Coluna de data identificada: %s", coluna_data)

    df[coluna_data] = converter_data(df[coluna_data])
    df = df.dropna(subset=[coluna_data]).set_index(coluna_data).sort_index()
    df.index.name = "data"

    # Mantém só as colunas conhecidas, converte para número e renomeia.
    faltando = [c for c in COLUNAS_CLIMA if c not in df.columns]
    if faltando:
        logger.warning("Colunas esperadas que não existem no CSV: %s", faltando)
    presentes = [c for c in COLUNAS_CLIMA if c in df.columns]
    extras = [c for c in df.columns if c not in COLUNAS_CLIMA]
    if extras:
        logger.warning("Colunas do CSV ignoradas (não estão em COLUNAS_CLIMA): %s", extras)

    df = df[presentes].apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns={c: COLUNAS_CLIMA[c]["nome"] for c in presentes})

    # Datas repetidas (ex.: várias regiões no mesmo arquivo) -> média por dia.
    repetidas = int(df.index.duplicated().sum())
    if repetidas:
        logger.warning(
            "%s datas repetidas encontradas; usando a média por dia. "
            "Se o CSV tiver várias regiões, o ideal é analisar cada uma separadamente.", repetidas
        )
        df = df.groupby(level=0).mean()

    inicio, fim = pd.Timestamp(GLOBAL_START_DATE), pd.Timestamp(GLOBAL_END_DATE)
    df = df.loc[(df.index >= inicio) & (df.index <= fim)].copy()
    logger.info("Clima após filtro %s a %s: %s", inicio.date(), fim.date(), df.shape)

    if df.empty:
        raise ValueError(
            f"Não há registros no período {GLOBAL_START_DATE} a {GLOBAL_END_DATE}. "
            "Confira as datas do CSV ou altere GLOBAL_START_DATE/GLOBAL_END_DATE."
        )
    return df


# ============================================================
# CARGA: CAFÉ (yfinance)
# ============================================================
def baixar_cafe():
    """Baixa o preço do café via yfinance (com cache local caso fique sem internet)."""
    cache = DATA_DIR / "cafe_yfinance_bruto.csv"
    fim_exclusivo = (pd.Timestamp(GLOBAL_END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        logger.info("Baixando %s via yfinance (%s a %s)", TICKER_CAFE, GLOBAL_START_DATE, GLOBAL_END_DATE)
        bruto = yf.download(
            TICKER_CAFE, start=GLOBAL_START_DATE, end=fim_exclusivo,
            auto_adjust=False, progress=False,
        )
        if bruto is None or bruto.empty:
            raise ValueError("yfinance retornou vazio.")
        if isinstance(bruto.columns, pd.MultiIndex):  # versões novas devolvem (campo, ticker)
            bruto.columns = bruto.columns.get_level_values(0)
        bruto.index = pd.to_datetime(bruto.index).tz_localize(None)
        bruto.index.name = "data"
        bruto.to_csv(cache, encoding="utf-8-sig")
        logger.info("Cache do café salvo: %s", cache)
    except Exception as erro:
        if cache.exists():
            logger.warning("Falha no yfinance (%s). Usando cache: %s", erro, cache)
            bruto = pd.read_csv(cache, index_col=0, parse_dates=True)
            bruto.index.name = "data"
        else:
            raise RuntimeError(f"Não foi possível baixar o café e não há cache: {erro}") from erro

    colunas = [c for c in COLUNAS_CAFE if c in bruto.columns]
    cafe = bruto[colunas].rename(columns={c: COLUNAS_CAFE[c]["nome"] for c in colunas}).sort_index()
    cafe = cafe.loc[(cafe.index >= GLOBAL_START_DATE) & (cafe.index <= GLOBAL_END_DATE)].dropna(subset=[COL_PRECO])
    cafe[COL_RETORNO] = cafe[COL_PRECO].pct_change() * 100
    logger.info("Café: %s pregões de %s a %s", len(cafe), cafe.index.min().date(), cafe.index.max().date())
    return cafe


# ============================================================
# INTEGRAÇÃO
# ============================================================
def agregar_mensal(clima, cafe):
    """Clima: média (temperatura, vento) ou soma (chuva, ETP). Café: média do fechamento."""
    agg = {v["nome"]: v["mensal"] for v in COLUNAS_CLIMA.values() if v["nome"] in clima.columns}
    clima_m = clima.resample("MS").agg(agg)
    # soma de mês sem nenhum dado vira 0; troca por NaN para não enganar
    contagem = clima.resample("MS").count()
    for c, how in agg.items():
        if how == "sum":
            clima_m.loc[contagem[c] == 0, c] = np.nan

    cafe_m = cafe[[COL_PRECO]].resample("MS").mean()
    mensal = clima_m.join(cafe_m, how="inner")
    mensal[COL_PRECO_VAR_MENSAL] = mensal[COL_PRECO].pct_change() * 100
    return mensal


def calcular_defasagens(mensal):
    """Correlação entre o clima defasado em k meses e o café (nível e variação mensal)."""
    nivel, variacao = {}, {}
    for c in NOMES_CLIMA:
        if c not in mensal.columns:
            continue
        nivel[c] = {f"{k}m": mensal[c].shift(k).corr(mensal[COL_PRECO]) for k in range(MAX_DEFASAGEM_MESES + 1)}
        variacao[c] = {f"{k}m": mensal[c].shift(k).corr(mensal[COL_PRECO_VAR_MENSAL])
                       for k in range(MAX_DEFASAGEM_MESES + 1)}
    return pd.DataFrame(nivel).T, pd.DataFrame(variacao).T


# ============================================================
# DICIONÁRIO DE DADOS
# ============================================================
def gerar_dicionario():
    linhas = []
    for orig, m in COLUNAS_CLIMA.items():
        linhas.append(dict(coluna_final=m["nome"], coluna_original=orig, fonte="CSV de clima (dados_brutos.csv)",
                           descricao=m["descricao"], unidade=m["unidade"],
                           agregacao_mensal="soma" if m["mensal"] == "sum" else "média"))
    for orig, m in COLUNAS_CAFE.items():
        linhas.append(dict(coluna_final=m["nome"], coluna_original=orig, fonte=f"yfinance ({TICKER_CAFE})",
                           descricao=m["descricao"], unidade=m["unidade"],
                           agregacao_mensal="média (fechamento)" if orig == "Close" else "-"))
    linhas.append(dict(coluna_final=COL_RETORNO, coluna_original="(calculada)", fonte="derivada do fechamento",
                       descricao="Variação percentual do fechamento em relação ao pregão anterior",
                       unidade="%", agregacao_mensal="-"))
    linhas.append(dict(coluna_final=COL_PRECO_VAR_MENSAL, coluna_original="(calculada)",
                       fonte="derivada do fechamento médio mensal",
                       descricao="Variação percentual do preço médio mensal vs. mês anterior",
                       unidade="%", agregacao_mensal="-"))
    return pd.DataFrame(linhas)


# ============================================================
# GRÁFICOS
# ============================================================
def grafico_series_clima(clima):
    for c in clima.columns:
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(clima.index, clima[c], linewidth=0.7, alpha=0.5, label="Diário")
            ax.plot(clima.index, clima[c].rolling(30, min_periods=10).mean(), linewidth=1.8,
                    label="Média móvel 30 dias")
            ax.set_title(f"Série temporal — {rot(c)}")
            ax.set_xlabel("Data")
            ax.set_ylabel(rot(c))
            ax.grid(True, alpha=0.25)
            ax.legend()
            salvar_fig(fig, "clima", f"serie_{c}.png")
        except Exception:
            logger.exception("Falha ao gerar série de %s", c)


def grafico_sazonalidade(clima):
    try:
        cols = list(clima.columns)
        n_rows = int(np.ceil(len(cols) / 3))
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.8 * n_rows))
        axes = np.array(axes).reshape(-1)
        meses = clima.index.month
        for ax, c in zip(axes, cols):
            sns.boxplot(x=meses, y=clima[c], ax=ax, color="#8ecae6", fliersize=2)
            ax.set_title(rot(c), fontsize=10)
            ax.set_xlabel("Mês")
            ax.set_ylabel("")
        for ax in axes[len(cols):]:
            ax.set_visible(False)
        fig.suptitle("Sazonalidade mensal do clima", y=1.0)
        fig.tight_layout()
        salvar_fig(fig, "clima", "sazonalidade_mensal.png")
    except Exception:
        logger.exception("Falha ao gerar sazonalidade.")


def grafico_cafe(cafe):
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(cafe.index, cafe[COL_PRECO], linewidth=1.2, label="Fechamento")
        ax1.plot(cafe.index, cafe[COL_PRECO].rolling(30, min_periods=10).mean(), linewidth=1.8,
                 label="Média móvel 30 pregões")
        ax1.set_title(f"Preço do café arábica ({TICKER_CAFE})")
        ax1.set_ylabel(rot(COL_PRECO))
        ax1.grid(True, alpha=0.25)
        ax1.legend()
        ax2.plot(cafe.index, cafe[COL_RETORNO], linewidth=0.6, color="gray")
        ax2.set_ylabel("Retorno diário (%)")
        ax2.set_xlabel("Data")
        ax2.grid(True, alpha=0.25)
        fig.tight_layout()
        salvar_fig(fig, "cafe", "serie_preco_cafe.png")
    except Exception:
        logger.exception("Falha ao gerar gráfico do café.")


def grafico_distribuicoes(completo, colunas):
    dados = completo[colunas]
    try:
        n_rows = int(np.ceil(len(colunas) / 3))
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.6 * n_rows))
        axes = np.array(axes).reshape(-1)
        for ax, c in zip(axes, colunas):
            ax.hist(dados[c].dropna(), bins=30, color="#219ebc", edgecolor="white")
            ax.set_title(rot(c), fontsize=10)
        for ax in axes[len(colunas):]:
            ax.set_visible(False)
        fig.suptitle("Distribuição das variáveis", y=1.0)
        fig.tight_layout()
        salvar_fig(fig, "distribuicoes", "histogramas.png")
    except Exception:
        logger.exception("Falha ao gerar histogramas.")

    try:
        n_rows = int(np.ceil(len(colunas) / 3))
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.6 * n_rows))
        axes = np.array(axes).reshape(-1)
        for ax, c in zip(axes, colunas):
            sns.boxplot(y=dados[c].dropna(), ax=ax, color="#8ecae6")
            ax.set_title(rot(c), fontsize=10)
            ax.set_ylabel("")
        for ax in axes[len(colunas):]:
            ax.set_visible(False)
        fig.suptitle("Boxplots (outliers)", y=1.0)
        fig.tight_layout()
        salvar_fig(fig, "distribuicoes", "boxplots.png")
    except Exception:
        logger.exception("Falha ao gerar boxplots.")


def grafico_heatmap(matriz, titulo, nome, grupo="correlacoes", vmin=-1, vmax=1, tamanho=None):
    try:
        fig, ax = plt.subplots(figsize=tamanho or (max(8, 2.1 * matriz.shape[1]), max(12, 1.8 * matriz.shape[0])))
        m = matriz.rename(index=rot, columns=lambda c: rot(c) if c in ROTULOS else c)
        sns.heatmap(m, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title(titulo)
        fig.tight_layout()
        salvar_fig(fig, grupo, nome)
    except Exception:
        logger.exception("Falha ao gerar heatmap %s", nome)


def grafico_clima_vs_cafe(mensal):
    for c in NOMES_CLIMA:
        if c not in mensal.columns:
            continue
        try:
            fig, ax1 = plt.subplots(figsize=(12, 4.5))
            ax1.plot(mensal.index, mensal[c], color="#2a9d8f", linewidth=1.6)
            ax1.set_ylabel(rot(c), color="#2a9d8f")
            ax2 = ax1.twinx()
            ax2.plot(mensal.index, mensal[COL_PRECO], color="#e76f51", linewidth=1.6)
            ax2.set_ylabel(rot(COL_PRECO) + " — média mensal", color="#e76f51")
            ax1.set_title(f"{rot(c)} (mensal) vs. preço do café")
            ax1.grid(True, alpha=0.25)
            fig.tight_layout()
            salvar_fig(fig, "clima_vs_cafe", f"mensal_{c}_vs_cafe.png")
        except Exception:
            logger.exception("Falha ao gerar clima vs café para %s", c)


# ============================================================
# RELATÓRIO
# ============================================================
def resumo_qualidade(df):
    out = pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "valores_ausentes": df.isna().sum(),
        "percentual_ausente": (df.isna().mean() * 100).round(2),
        "valores_unicos": df.nunique(dropna=True),
    })
    out.index.name = "coluna"
    return out


def estatisticas_descritivas(df):
    est = df.describe().T
    est["mediana"] = df.median()
    est["assimetria"] = df.skew()
    est["variancia"] = df.var()
    est.index.name = "coluna"
    return est


def gerar_relatorio(clima, cafe, completo, pregao, mensal, qual, est, corr, corr_cafe, lag_nivel, lag_var):
    r = []
    r.append("RELATÓRIO DE ANÁLISE EXPLORATÓRIA — IC_CAFE (CLIMA + CAFÉ ARÁBICA)")
    r.append("=" * 70)
    r.append(f"Arquivo de clima: {PATH_RAW_DATA}")
    r.append(f"Café: yfinance, ticker {TICKER_CAFE} (¢US$/libra-peso)")
    r.append(f"Período configurado: {GLOBAL_START_DATE} a {GLOBAL_END_DATE}")
    r.append("")
    r.append("COBERTURA DOS DADOS")
    r.append(f"  Clima: {len(clima)} dias ({clima.index.min().date()} a {clima.index.max().date()})")
    r.append(f"  Café: {len(cafe)} pregões ({cafe.index.min().date()} a {cafe.index.max().date()})")
    r.append(f"  Dias com clima E café (base diária integrada): {len(pregao)}")
    r.append(f"  Meses na base mensal integrada: {len(mensal)}")
    r.append(f"  Células ausentes (clima): {int(clima.isna().sum().sum())}")
    r.append("")
    r.append("QUALIDADE DOS DADOS (clima + café)")
    r.append(qual.to_string())
    r.append("")
    r.append("ESTATÍSTICAS DESCRITIVAS")
    r.append(est.round(4).to_string())
    r.append("")
    r.append("CORRELAÇÃO DE PEARSON (dias de pregão)")
    r.append(corr.round(4).to_string())
    r.append("")
    r.append(f"CORRELAÇÕES COM {COL_PRECO} (ordenadas pelo valor absoluto)")
    r.append(corr_cafe.round(4).to_string())
    r.append("")
    r.append(f"DEFASAGENS (mensal): clima defasado em k meses vs. {COL_PRECO}")
    r.append(lag_nivel.round(4).to_string())
    r.append("")
    r.append(f"DEFASAGENS (mensal): clima defasado em k meses vs. {COL_PRECO_VAR_MENSAL}")
    r.append(lag_var.round(4).to_string())
    r.append("")
    r.append("OBSERVAÇÕES")
    r.append("  - Correlação com o NÍVEL do preço pode ser espúria (séries com tendência);")
    r.append("    a correlação com a VARIAÇÃO mensal é mais confiável para sinal de curto prazo.")
    r.append("  - Chuva e ETP são somadas no mês; temperaturas e vento são médias.")
    r.append("  - O contínuo KC=F troca de vencimento ao longo do tempo (pequenos saltos).")
    caminho = REPORT_DIR / "relatorio_eda.txt"
    caminho.write_text("\n".join(r), encoding="utf-8")
    logger.info("Relatório salvo: %s", caminho)


# ============================================================
# PIPELINE
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("INÍCIO DA ANÁLISE DO IC_CAFE")
    try:
        clima = carregar_clima(PATH_RAW_DATA)
        cafe = baixar_cafe()

        completo = clima.join(cafe, how="left")        # todos os dias de clima; café só nos pregões
        pregao = clima.join(cafe, how="inner")         # só dias com clima e café
        mensal = agregar_mensal(clima, cafe)

        # --- Bases limpas ---
        salvar_tabela(clima, "clima_diario.csv", DATA_DIR)
        salvar_tabela(cafe, "cafe_diario.csv", DATA_DIR)
        salvar_tabela(completo, "integrado_diario_completo.csv", DATA_DIR)
        salvar_tabela(pregao, "integrado_diario_pregoes.csv", DATA_DIR)
        salvar_tabela(mensal, "integrado_mensal.csv", DATA_DIR)

        # --- Dicionário ---
        salvar_tabela(gerar_dicionario(), "dicionario_de_dados.csv", REPORT_DIR, indice=False)

        # --- Tabelas ---
        cols_analise = [c for c in NOMES_CLIMA if c in pregao.columns] + [COL_PRECO, COL_RETORNO]
        qual = resumo_qualidade(completo)
        est = estatisticas_descritivas(completo[cols_analise + [c for c in completo.columns if c not in cols_analise]])
        corr = pregao[cols_analise].corr(method="pearson")
        corr_sp = pregao[cols_analise].corr(method="spearman")
        corr_cafe = (corr[COL_PRECO].drop(labels=[COL_PRECO])
                     .sort_values(key=lambda s: s.abs(), ascending=False))
        corr_cafe.index.name = "variavel"
        lag_nivel, lag_var = calcular_defasagens(mensal)
        lag_nivel.index.name = lag_var.index.name = "variavel_climatica"

        salvar_tabela(qual, "qualidade_dos_dados.csv")
        salvar_tabela(est, "estatisticas_descritivas.csv")
        salvar_tabela(corr, "correlacao_pearson.csv")
        salvar_tabela(corr_sp, "correlacao_spearman.csv")
        salvar_tabela(corr_cafe.to_frame("correlacao_com_cafe"), "correlacao_com_cafe.csv")
        salvar_tabela(lag_nivel, "defasagens_clima_vs_cafe_nivel.csv")
        salvar_tabela(lag_var, "defasagens_clima_vs_cafe_variacao.csv")

        # --- Relatório ---
        gerar_relatorio(clima, cafe, completo, pregao, mensal, qual, est, corr, corr_cafe, lag_nivel, lag_var)

        # --- Gráficos ---
        grafico_series_clima(clima)
        grafico_sazonalidade(clima)
        grafico_cafe(cafe)
        grafico_distribuicoes(completo, cols_analise)
        grafico_heatmap(corr, "Correlação de Pearson (dias de pregão)", "heatmap_pearson.png")
        grafico_heatmap(corr_sp, "Correlação de Spearman (dias de pregão)", "heatmap_spearman.png")
        grafico_heatmap(lag_nivel, f"Clima defasado vs. {rot(COL_PRECO)} (mensal)",
                        "heatmap_defasagem_nivel.png", tamanho=(9, 5))
        grafico_heatmap(lag_var, f"Clima defasado vs. {rot(COL_PRECO_VAR_MENSAL)} (mensal)",
                        "heatmap_defasagem_variacao.png", tamanho=(9, 5))
        grafico_clima_vs_cafe(mensal)

        logger.info("EXECUÇÃO FINALIZADA COM SUCESSO. Resultados em: %s", RESULTS_DIR)
    except Exception as erro:
        logger.exception("Falha na execução: %s", erro)
        raise


if __name__ == "__main__":
    main()