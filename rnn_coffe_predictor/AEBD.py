import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import logging

# =========================
# CONFIGURAÇÃO DE LOGGING
# =========================
LOG_DIR = r"C:\Users\joao3\Desktop\IC_Cafe\logs"
LOG_FILE = os.path.join(LOG_DIR, "eda_analysis.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# DEFINIÇÃO DO PERÍODO GLOBAL DE ANÁLISE
# AJUSTADO PARA O NOVO DATASET CLIMÁTICO (1975-2015)
# =========================
GLOBAL_START_DATE = '1975-01-01'
GLOBAL_END_DATE = '2015-12-31'

# =========================
# 1. COLETA E PRÉ-PROCESSAMENTO
# =========================
def baixar_e_processar_acao(ticker, nome_coluna_final, start=GLOBAL_START_DATE, end=GLOBAL_END_DATE):
    logger.info(f"Iniciando download e processamento de dados para {nome_coluna_final} ({ticker}) de {start} a {end}...")
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)

        if df.empty:
            logger.error(f"Erro: Dados para {nome_coluna_final} ({ticker}) não foram baixados corretamente ou o DataFrame está vazio para o período {start} a {end}.")
            raise ValueError(f"Erro: Dados para {nome_coluna_final} ({ticker}) não foram baixados corretamente para o período.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) and col[1] != '' else col[0] for col in df.columns.values]
            logger.info(f"MultiIndex detectado e achatado para {nome_coluna_final} ({ticker}). Novas colunas: {df.columns.tolist()}")

        logger.info(f"Colunas disponíveis para {nome_coluna_final} ({ticker}) após download e possível achatamento: {df.columns.tolist()}")

        coluna_para_selecionar = None
        expected_close_col_name = f'Close_{ticker}'
        simple_close_col_name = 'Close'

        if expected_close_col_name in df.columns:
            coluna_para_selecionar = expected_close_col_name
        elif simple_close_col_name in df.columns:
            coluna_para_selecionar = simple_close_col_name
        else:
            for col in df.columns:
                if (isinstance(col, tuple) and col[0] == 'Close') or (isinstance(col, str) and col == 'Close'):
                    coluna_para_selecionar = col
                    break
            
            if coluna_para_selecionar is None:
                logger.error(f"Erro: Coluna de fechamento ('{expected_close_col_name}' ou '{simple_close_col_name}' ou tupla 'Close') não encontrada no DataFrame para {nome_coluna_final} ({ticker}). Colunas disponíveis: {df.columns.tolist()}")
                raise KeyError(f"Coluna de fechamento não encontrada no DataFrame para {nome_coluna_final}.")

        logger.info(f"Usando a coluna '{coluna_para_selecionar}' para {nome_coluna_final}.")

        df = df[[coluna_para_selecionar]]
        df.rename(columns={coluna_para_selecionar: nome_coluna_final}, inplace=True)

        df.index.name = 'Date'
        logger.info(f"[OK] Dados de {nome_coluna_final} processados com colunas: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Exceção ao baixar/processar dados de {nome_coluna_final}: {e}", exc_info=True)
        raise

# As funções de download de commodities agora usam as datas globais por padrão
def baixar_e_processar_cafe(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE):
    return baixar_e_processar_acao('KC=F', 'PrecoCafe', start, end)

def baixar_e_processar_petroleo(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE):
    return baixar_e_processar_acao('CL=F', 'PrecoPetroleo', start, end)

def baixar_e_processar_milho(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE):
    return baixar_e_processar_acao('ZC=F', 'PrecoMilho', start, end)

def baixar_e_processar_clima(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE):
    logger.info("Iniciando download e processamento de dados climáticos 'rogerioifpr/dados-climticos-de-1975-2015' via kagglehub...")
    try:
        path = kagglehub.dataset_download("rogerioifpr/dados-climticos-de-1975-2015")
        logger.info(f"[OK] Arquivos baixados em: {path}")

        arquivos_csv = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not arquivos_csv:
            logger.error(f"Erro: Nenhum arquivo CSV encontrado na pasta do dataset KaggleHub: {path}.")
            raise FileNotFoundError("Erro: Nenhum arquivo CSV encontrado no dataset climático.")
        
        caminho_csv = os.path.join(path, arquivos_csv[0])
        logger.info(f"Lendo arquivo climático: {caminho_csv}")
        df = pd.read_csv(caminho_csv)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else str(col) for col in df.columns.values]
            logger.info(f"MultiIndex detectado e achatado para dados climáticos. Novas colunas: {df.columns.tolist()}")
        else:
            logger.info("Nenhum MultiIndex detectado nas colunas do dataset climático. Colunas atuais: %s", df.columns.tolist())

        # Criar a coluna 'Date' a partir de 'ano' e 'mes', garantindo o dia 1
        df['Date'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01', errors='coerce')
        df = df.set_index('Date')
        
        df_clima_filtrado = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))].copy()

        column_mapping = {
            'chuva': 'ChuvaMM',
            'evaporacao': 'EvaporacaoMM',
            'insolacao': 'InsolacaoHoras',
            'tempmed': 'TemperaturaMediaC',
            'umidrel': 'UmidadeRelativa%',
            'tempmax_abs': 'TemperaturaMaxAbsC',
            'tempmax_med': 'TemperaturaMaxMedC',
            'tempmin_abs': 'TemperaturaMinAbsC',
            'tempmin_med': 'TemperaturaMinMedC'
        }
        actual_column_mapping = {k: v for k, v in column_mapping.items() if k in df_clima_filtrado.columns}
        df_clima_filtrado.rename(columns=actual_column_mapping, inplace=True)

        for col_final in actual_column_mapping.values():
            df_clima_filtrado[col_final] = pd.to_numeric(df_clima_filtrado[col_final], errors='coerce')
        
        df_clima_filtrado = df_clima_filtrado.interpolate(method='linear', limit_direction='both')
        df_clima_filtrado = df_clima_filtrado.dropna()

        df_clima_filtrado = df_clima_filtrado[list(actual_column_mapping.values())]

        df_clima_filtrado.index.name = 'Date'
        logger.info("[OK] Dados climáticos processados com colunas: %s", df_clima_filtrado.columns.tolist())
        
        if df_clima_filtrado.empty:
            logger.warning(f"DataFrame climático vazio após processamento e filtragem para o período {start} a {end}.")
            return pd.DataFrame(columns=list(actual_column_mapping.values()), index=pd.to_datetime([])).set_index('Date')

        return df_clima_filtrado
    except Exception as e:
        logger.error(f"Exceção ao baixar/processar dados climáticos: {e}", exc_info=True)
        raise

# =========================
# Função para normalizar data dos preços para o primeiro dia do mês
# =========================
def normaliza_data_mensal(df_para_normalizar, df_nome=""):
    logger.info(f"Normalizando índice do dataframe {df_nome} para o primeiro dia do mês...")
    df = df_para_normalizar.copy()
    try:
        df.index = pd.to_datetime(df.index)
        df = df[(df.index >= pd.to_datetime(GLOBAL_START_DATE)) & (df.index <= pd.to_datetime(GLOBAL_END_DATE))]

        if df.empty:
            logger.warning(f"DataFrame {df_nome} ficou vazio após filtragem pelo período global. Retornando DataFrame vazio para evitar erros de merge.")
            return pd.DataFrame(columns=df_para_normalizar.columns, index=pd.to_datetime([])).set_index('Date')

        # --- CORREÇÃO FINAL AQUI: Garante que o dia seja sempre o primeiro do mês ---
        # Converte para período mensal (YYYY-MM) e depois para timestamp com o dia 1
        df.index = df.index.to_period('M').to_timestamp(how='start') # 'how=start' é explícito para o início do mês
        
        df = df.groupby(df.index).mean() # Média se houver vários registros no mês
        df.index.name = 'Date'
        logger.info(f"[OK] Normalização do dataframe {df_nome} concluída. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao normalizar datas do dataframe {df_nome}: {e}", exc_info=True)
        raise

# =========================
# 2. UNIÃO DOS DADOS
# =========================
def unir_dados(df_cafe, df_clima, df_petroleo, df_milho):
    logger.info("Iniciando união dos dados de café, clima, petróleo e milho...")
    try:
        df_cafe_norm = normaliza_data_mensal(df_cafe, "Café")
        df_petroleo_norm = normaliza_data_mensal(df_petroleo, "Petróleo")
        df_milho_norm = normaliza_data_mensal(df_milho, "Milho")

        dataframes_para_unir = {
            'Cafe': df_cafe_norm,
            'Clima': df_clima,
            'Petroleo': df_petroleo_norm,
            'Milho': df_milho_norm
        }

        min_date = pd.Timestamp.min
        max_date = pd.Timestamp.max

        for nome, df_current in dataframes_para_unir.items():
            if not df_current.empty:
                current_min = df_current.index.min()
                current_max = df_current.index.max()
                logger.info(f"Datas de '{nome}': {current_min} a {current_max}")
                if current_min > min_date:
                    min_date = current_min
                if current_max < max_date:
                    max_date = current_max
            else:
                logger.warning(f"DataFrame '{nome}' está vazio. Isso pode levar a um merge vazio.")
                min_date = pd.Timestamp.max
                max_date = pd.Timestamp.min
                break

        if min_date > max_date:
            logger.warning("Não há período de sobreposição válido entre todos os DataFrames. O DataFrame final estará vazio.")
            all_cols = []
            for df_col in [df_cafe_norm.columns, df_clima.columns, df_petroleo_norm.columns, df_milho_norm.columns]:
                all_cols.extend(df_col.tolist())
            return pd.DataFrame(columns=list(set(all_cols)), index=pd.to_datetime([])).set_index('Date')

        logger.info(f"Período de sobreposição comum encontrado: {min_date} a {max_date}")

        # Começamos com o DataFrame de café já filtrado pelo período comum
        df_completo = dataframes_para_unir['Cafe'][(dataframes_para_unir['Cafe'].index >= min_date) & (dataframes_para_unir['Cafe'].index <= max_date)]
        
        if df_completo.empty:
            logger.warning("DataFrame de Café ficou vazio após filtragem pelo período de sobreposição. O DataFrame final estará vazio.")
            all_cols = []
            for df_col in [df_cafe_norm.columns, df_clima.columns, df_petroleo_norm.columns, df_milho_norm.columns]:
                all_cols.extend(df_col.tolist())
            return pd.DataFrame(columns=list(set(all_cols)), index=pd.to_datetime([])).set_index('Date')

        logger.info(f"Iniciando merge com Café (filtrado pelo período comum). Shape inicial: {df_completo.shape}")

        for nome, df_to_merge in dataframes_para_unir.items():
            if nome != 'Cafe':
                if not df_to_merge.empty:
                    df_to_merge_filtered = df_to_merge[(df_to_merge.index >= min_date) & (df_to_merge.index <= max_date)]
                    
                    df_completo = pd.merge(
                        df_completo.reset_index(),
                        df_to_merge_filtered.reset_index(),
                        on='Date',
                        how='inner'
                    )
                    df_completo.set_index('Date', inplace=True)
                    logger.info(f"Merge com {nome} concluído. Shape: {df_completo.shape}")
                else:
                    logger.warning(f"DataFrame '{nome}' está vazio, pulando o merge. O resultado final será vazio.")
                    all_cols = []
                    for df_col in [df_cafe_norm.columns, df_clima.columns, df_petroleo_norm.columns, df_milho_norm.columns]:
                        all_cols.extend(df_col.tolist())
                    return pd.DataFrame(columns=list(set(all_cols)), index=pd.to_datetime([])).set_index('Date')

        df_completo.index = pd.to_datetime(df_completo.index)
        logger.info(f"[OK] Dados unidos com sucesso! Shape final: {df_completo.shape}")
        return df_completo
    except Exception as e:
        logger.error(f"Erro ao unir os dataframes: {e}", exc_info=True)
        raise

# =========================
# 3. ANÁLISE EXPLORATÓRIA
# =========================
def realizar_eda(df_final):
    logger.info("Iniciando Análise Exploratória de Dados (EDA)...")
    try:
        os.makedirs('graficos_eda', exist_ok=True)
        logger.info("Diretório 'graficos_eda' verificado/criado.")

        logger.info("\n=== ESTATÍSTICAS DESCRITIVAS ===")
        if not df_final.empty:
            logger.info("\n%s", df_final.describe().to_string())
        else:
            logger.warning("DataFrame final vazio. Não é possível gerar estatísticas descritivas.")

        df_plot = df_final.select_dtypes(include=np.number)
        if df_plot.empty:
            logger.warning("DataFrame para plotagem está vazio após selecionar colunas numéricas. NENHUM GRÁFICO SERÁ GERADO.")
            return

        logger.info("Gerando gráficos de série temporal...")
        for coluna in df_plot.columns:
            try:
                df_plot[coluna].plot(title=f'Série Temporal de {coluna}', figsize=(12, 4))
                plt.ylabel(coluna)
                plt.tight_layout()
                plt.savefig(f'graficos_eda/serie_{coluna}.png')
                plt.close()
                logger.info(f"Gráfico de série temporal para '{coluna}' salvo.")
            except Exception as e:
                logger.error(f"Erro ao gerar gráfico de série temporal para '{coluna}': {e}", exc_info=True)

        logger.info("Gerando histogramas...")
        try:
            df_plot.hist(bins=30, figsize=(12, 8))
            plt.tight_layout()
            plt.savefig('graficos_eda/histogramas.png')
            plt.close()
            logger.info("Histogramas salvos.")
        except Exception as e:
            logger.error(f"Erro ao gerar histogramas: {e}", exc_info=True)

        logger.info("Gerando boxplots...")
        try:
            n_cols = len(df_plot.columns)
            n_rows = int(np.ceil(n_cols / 3))
            df_plot.plot(kind='box', subplots=True, layout=(n_rows, 3), figsize=(12, n_rows * 4))
            plt.tight_layout()
            plt.savefig('graficos_eda/boxplots.png')
            plt.close()
            logger.info("Boxplots salvos.")
        except Exception as e:
            logger.error(f"Erro ao gerar boxplots: {e}", exc_info=True)

        logger.info("Calculando matriz de correlação...")
        corr = df_plot.corr()
        logger.info("\n=== MATRIZ DE CORRELAÇÃO ===")
        logger.info("\n%s", corr.to_string())

        if 'PrecoCafe' in corr.columns:
            corrs_com_cafe = corr['PrecoCafe'].drop('PrecoCafe', errors='ignore').sort_values(key=lambda x: abs(x), ascending=False)
            logger.info("\n=== CORRELAÇÕES COM O PREÇO DO CAFÉ ===")
            logger.info("\n%s", corrs_com_cafe.to_string())
        else:
            logger.warning("'PrecoCafe' não encontrado nas colunas para cálculo de correlação detalhada.")

        logger.info("Gerando heatmap da correlação...")
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Matriz de Correlação')
            plt.tight_layout()
            plt.savefig('graficos_eda/heatmap_correlacoes.png')
            plt.close()
            logger.info("Heatmap de correlação salvo.")
        except Exception as e:
            logger.error(f"Erro ao gerar heatmap de correlação: {e}", exc_info=True)

        logger.info("Análise Exploratória de Dados (EDA) concluída.")
    except Exception as e:
        logger.error(f"Exceção geral durante a EDA: {e}", exc_info=True)
        raise

# =========================
# MAIN SCRIPT
# =========================
if __name__ == '__main__':
    logger.info("===== INICIANDO EXECUÇÃO PRINCIPAL DO SCRIPT =====")
    try:
        df_cafe = baixar_e_processar_acao('KC=F', 'PrecoCafe')
        df_petroleo = baixar_e_processar_acao('CL=F', 'PrecoPetroleo')
        df_milho = baixar_e_processar_acao('ZC=F', 'PrecoMilho')

        df_clima = baixar_e_processar_clima(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE)

        logger.info("Primeiras e últimas datas do café (dados brutos): %s -> %s", df_cafe.index.min(), df_cafe.index.max())
        logger.info("Primeiras e últimas datas do petróleo (dados brutos): %s -> %s", df_petroleo.index.min(), df_petroleo.index.max())
        logger.info("Primeiras e últimas datas do milho (dados brutos): %s -> %s", df_milho.index.min(), df_milho.index.max())
        logger.info("Primeiras e últimas datas do clima (dados brutos): %s -> %s", df_clima.index.min(), df_clima.index.max())

        df_final = unir_dados(df_cafe, df_clima, df_petroleo, df_milho)

        logger.info("Colunas do DataFrame Final: %s", df_final.columns.tolist())
        logger.info("Primeiras linhas do DataFrame Final:\n%s", df_final.head().to_string())
        logger.info("Últimas linhas do DataFrame Final:\n%s", df_final.tail().to_string())

        realizar_eda(df_final)
        logger.info("===== EXECUÇÃO PRINCIPAL DO SCRIPT CONCLUÍDA COM SUCESSO =====")

    except Exception as e:
        logger.critical(f"Erro crítico na execução principal do script: {e}", exc_info=True)