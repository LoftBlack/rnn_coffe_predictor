import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
KAGGLE_DATASET_PATH = r"C:\Users\joao3\Desktop\IC_Cafe\rnn_coffe_predictor\datasets"
GLOBAL_START_DATE = "1990-01-01"
GLOBAL_END_DATE = "2023-12-31"
COFFEE_FUTURES_SYMBOL = "KC=F"
GRAPHICS_DIR = "graficos_eda_final_2"
LOG_PATH = r"C:\Users\joao3\Desktop\IC_Cafe\logs\analise_cafe_completa_2.log"
TAMANHO_IMAGEM = (24, 16)

# Logging Configuration
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(GRAPHICS_DIR, exist_ok=True)
logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_datasets(path):
    dataframes = {}
    try:
        if not os.path.exists(path):
            logging.error(f"O caminho especificado não existe: {path}")
            return dataframes

        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not files:
            logging.error("Nenhum arquivo CSV encontrado no diretório.")
            return dataframes

        for file in files:
            df_name = file.split('.')[0]
            dataframes[df_name] = pd.read_csv(os.path.join(path, file))
            logging.info(f"Arquivo {file} carregado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao carregar arquivos: {e}")
    return dataframes

def transform_and_aggregate(dataframes):
    master_df = pd.DataFrame()
    for name, df in dataframes.items():
        try:
            id_vars = df.columns[0:1].tolist()
            year_cols = df.columns[1:].tolist()
            long_df = df.melt(id_vars=id_vars, var_name='Ano', value_name=f'Valor{name}')
            long_df['Ano'] = long_df['Ano'].astype(int)
            aggregated = long_df.groupby('Ano')[f'Valor{name}'].mean().reset_index()
            aggregated.rename(columns={f'Valor{name}': f'{name}_MediaAnual'}, inplace=True)

            if master_df.empty:
                master_df = aggregated
            else:
                master_df = master_df.merge(aggregated, on='Ano', how='outer')
            logging.info(f"Transformação e agregação para {name} concluídas.")
        except Exception as e:
            logging.error(f"Erro ao transformar e agregar {name}: {e}")
    logging.info(f"Master DataFrame após agregação: {master_df.head()}")
    return master_df

def fetch_coffee_prices(start_date, end_date):
    try:
        coffee_data = yf.download(COFFEE_FUTURES_SYMBOL, start=start_date, end=end_date, group_by="ticker")
        
        # Remove MultiIndex de qualquer profundidade
        coffee_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in coffee_data.columns]

        coffee_data = coffee_data.reset_index()
        coffee_data['Year'] = pd.to_datetime(coffee_data['Date']).dt.year

        annual_avg = coffee_data.groupby('Year', as_index=False)['Close'].mean()
        annual_avg.rename(columns={'Close': 'PrecoCafe_MediaAnual'}, inplace=True)

        logging.info("Média anual do preço do café calculada com sucesso.")
        return annual_avg
    except Exception as e:
        logging.error(f"Erro ao obter dados de preços de café: {e}")
        return pd.DataFrame()

def merge_data(master_df, coffee_price_df):
    merged_df = master_df.merge(coffee_price_df, left_on='Ano', right_on='Year', how='outer')
    merged_df.drop(columns='Year', inplace=True)
    logging.info("Dados mesclados com sucesso.")
    return merged_df

def perform_eda(df):
    logging.info("Gerando estatísticas descritivas.")
    print(df.describe())

    logging.info("Calculando a matriz de correlação.")
    correlation_matrix = df.drop(columns='Ano').corr()
    plt.figure(figsize=TAMANHO_IMAGEM)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Matriz de Correlação')
    plt.savefig(os.path.join(GRAPHICS_DIR, 'matriz_correlacao.png'))
    plt.close()

    for column in df.columns[1:]:
        plt.figure(figsize=TAMANHO_IMAGEM)
        plt.plot(df['Ano'], df[column], marker='o')
        plt.title(f'Evolução de {column}')
        plt.xlabel('Ano')
        plt.ylabel(column)
        plt.grid()
        plt.savefig(os.path.join(GRAPHICS_DIR, f'evolucao_{column}.png'))
        plt.close()

    if 'PrecoCafe_MediaAnual' in df.columns and 'disappearance_MediaAnual' in df.columns:
        plt.figure(figsize=TAMANHO_IMAGEM)
        plt.plot(df['Ano'], df['PrecoCafe_MediaAnual'], label='Preço Café (Média Anual)', color='orange')
        plt.plot(df['Ano'], df['disappearance_MediaAnual'], label='Desaparecimento Média Anual', color='blue')
        plt.title('Preço Médio do Café e Desaparecimento')
        plt.xlabel('Ano')
        plt.ylabel('Valores')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(GRAPHICS_DIR, 'overlay_preco_desaparecimento.png'))
        plt.close()

    for column in df.columns[1:]:
        plt.figure(figsize=TAMANHO_IMAGEM)
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribuição de {column}')
        plt.xlabel(column)
        plt.ylabel('Frequência')
        plt.savefig(os.path.join(GRAPHICS_DIR, f'distribuicao_{column}.png'))
        plt.close()

    for column in df.columns[1:]:
        if column != 'PrecoCafe_MediaAnual':
            plt.figure(figsize=TAMANHO_IMAGEM)
            sns.scatterplot(x=df[column], y=df['PrecoCafe_MediaAnual'])
            sns.regplot(x=df[column], y=df['PrecoCafe_MediaAnual'], scatter=False, color='red')
            plt.title(f'Relação entre {column} e Preço Médio do Café')
            plt.xlabel(column)
            plt.ylabel('Preço Médio do Café')
            plt.savefig(os.path.join(GRAPHICS_DIR, f'relacao_{column}_preco.png'))
            plt.close()

    selected_columns = df.drop(columns='Ano').columns.tolist()[:5]
    if 'PrecoCafe_MediaAnual' not in selected_columns:
        selected_columns = ['PrecoCafe_MediaAnual'] + selected_columns[:4]
    sns.pairplot(df[selected_columns])
    plt.savefig(os.path.join(GRAPHICS_DIR, 'pairplot.png'))
    plt.close()

def main():
    logging.info("Iniciando o processo de análise de dados de café.")
    dataframes = load_datasets(KAGGLE_DATASET_PATH)
    master_df = transform_and_aggregate(dataframes)
    coffee_price_df = fetch_coffee_prices(GLOBAL_START_DATE, GLOBAL_END_DATE)
    print("\n--- DEBUG MASTER_DF ---")
    print(master_df.head())
    print(master_df.columns)
    print(master_df.index)
    print(master_df.dtypes)
    print("\n--- DEBUG COFFEE_PRICE_DF ---")
    print(coffee_price_df.head())
    print(coffee_price_df.columns)
    print(coffee_price_df.index)
    print(coffee_price_df.dtypes)

    final_df = merge_data(master_df, coffee_price_df)
    perform_eda(final_df)
    logging.info("Análise de dados de café concluída com sucesso.")

if __name__ == "__main__":
    main()
