# analise_cafe_cambio.py

import os
import logging
from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações iniciais de estilo
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.figsize'] = (14, 6)

# Caminhos de saída
LOG_PATH = r"C:\Users\joao3\Desktop\IC_Cafe\logs"
GRAFICOS_PATH = r"C:\Users\joao3\Desktop\IC_Cafe\graficos_AECF"
LOG_FILE = os.path.join(LOG_PATH, "analise_cafe_cambio.log")

# ========================
# Funções de Utilidade
# ========================
def configurar_logging():
    os.makedirs(LOG_PATH, exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Logging configurado com sucesso.")


def salvar_grafico(fig, nome):
    os.makedirs(GRAFICOS_PATH, exist_ok=True)
    caminho = os.path.join(GRAFICOS_PATH, nome)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)
    logging.info(f"Gráfico salvo em: {caminho}")

# ========================
# Aquisição de Dados
# ========================
def baixar_dados(tickers, inicio, fim):
    dados = {}
    for ticker, nome_coluna in tickers.items():
        try:
            df = yf.download(ticker, start=inicio, end=fim)[['Close']]
            df.rename(columns={'Close': nome_coluna}, inplace=True)
            dados[nome_coluna] = df
            logging.info(f"Dados baixados com sucesso para {ticker}")
        except Exception as e:
            logging.error(f"Erro ao baixar dados de {ticker}: {e}")
    return dados

# ========================
# Pré-processamento
# ========================
def preprocessar_dados(dados_dict):
    df_merged = pd.concat(dados_dict.values(), axis=1, join='inner')
    df_merged.dropna(inplace=True)
    logging.info(f"DataFrame mesclado com shape: {df_merged.shape}")

    # Calcular retornos
    retornos = df_merged.pct_change().rename(columns={
    'PrecoCafe': 'RetornoPrecoCafe',
    'BRL_USD': 'RetornoBRL_USD',
    'EUR_USD': 'RetornoEUR_USD',
    'EUR_BRL': 'RetornoEUR_BRL'
    })
    df_merged = pd.concat([df_merged, retornos], axis=1).dropna()

    df_merged.dropna(inplace=True)
    logging.info("Retornos calculados e NaNs removidos.")
    return df_merged

# ========================
# Visualizações
# ========================
def plot_series_temporais(df):
    for coluna in ['PrecoCafe', 'BRL_USD', 'EUR_USD', 'EUR_BRL']:
        fig, ax = plt.subplots()
        df[coluna].plot(ax=ax)
        ax.set_title(f"Série Temporal de {coluna}")
        ax.set_ylabel("Valor")
        salvar_grafico(fig, f"serie_{coluna}.png")

    for coluna in ['RetornoPrecoCafe', 'RetornoBRL_USD', 'RetornoEUR_USD', 'RetornoEUR_BRL']:
        fig, ax = plt.subplots()
        df[coluna].plot(ax=ax)
        ax.set_title(f"Retorno Diário de {coluna}")
        ax.set_ylabel("Retorno")
        salvar_grafico(fig, f"retorno_{coluna}.png")

    # Eixos duplos
    pares = ['BRL_USD', 'EUR_USD', 'EUR_BRL']
    for par in pares:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        df['PrecoCafe'].plot(ax=ax1, color='tab:brown', label='Café')
        df[par].plot(ax=ax2, color='tab:blue', label=par)
        ax1.set_title(f"Preço do Café vs {par}")
        ax1.set_ylabel("Preço Café")
        ax2.set_ylabel(par)
        salvar_grafico(fig, f"comparacao_{par}.png")

# ========================
# Estatísticas e Correlação
# ========================
def analise_estatistica(df):
    desc = df.describe()
    logging.info(f"Estatísticas descritivas:\n{desc}")
    return desc

def analise_correlacao(df):
    retornos = df.filter(like='Retorno')
    corr = retornos.corr()

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title("Matriz de Correlação dos Retornos")
    salvar_grafico(fig, "heatmap_correlacao.png")

    # Tabela de correlação com Café
    cafe_corr = corr['RetornoPrecoCafe'].drop('RetornoPrecoCafe').sort_values(key=abs, ascending=False)
    logging.info("Correlação com Retorno do Café:\n" + str(cafe_corr))

    # Gráficos de dispersão
    for col in cafe_corr.index:
        fig, ax = plt.subplots()
        sns.regplot(x=df['RetornoPrecoCafe'], y=df[col], ax=ax)
        ax.set_title(f"Dispersão: Retorno do Café vs {col}")
        salvar_grafico(fig, f"dispersao_cafe_vs_{col}.png")

    return corr, cafe_corr

def plot_distribuicoes(df):
    for col in ['PrecoCafe', 'BRL_USD', 'EUR_USD', 'EUR_BRL']:
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=50, kde=True, ax=ax)
        ax.set_title(f"Histograma de {col}")
        salvar_grafico(fig, f"histograma_{col}.png")

    for col in df.columns:
        if "Retorno" in col:
            fig, ax = plt.subplots()
            sns.violinplot(y=df[col], ax=ax)
            ax.set_title(f"Distribuição de {col}")
            salvar_grafico(fig, f"violin_{col}.png")

    # Pairplot
    subset = df[["RetornoPrecoCafe", "RetornoBRL_USD", "RetornoEUR_USD", "RetornoEUR_BRL"]]
    fig = sns.pairplot(subset)
    fig.fig.suptitle("Pairplot dos Retornos", y=1.02)
    fig.savefig(os.path.join(GRAFICOS_PATH, "pairplot_retornos.png"))
    plt.close()

# ========================
# Execução Principal
# ========================

def main():
    configurar_logging()
    inicio = '2000-01-01'
    fim = datetime.today().strftime('%Y-%m-%d')

    tickers = {
        'KC=F': 'PrecoCafe',
        'BRL=X': 'BRL_USD',
        'EURUSD=X': 'EUR_USD',
        'EURBRL=X': 'EUR_BRL'
    }

    dados = baixar_dados(tickers, inicio, fim)
    df = preprocessar_dados(dados)

    print("\nEstatísticas Descritivas:\n")
    print(analise_estatistica(df))

    plot_series_temporais(df)
    plot_distribuicoes(df)

    _, cafe_corr = analise_correlacao(df)
    print("\nCorrelação dos Retornos com o Café:\n")
    print(cafe_corr)

if __name__ == "__main__":
    main()
