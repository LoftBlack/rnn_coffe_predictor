import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
KAGGLE_DATASET_PATH = r"C:\Users\joao3\Desktop\IC_Cafe\rnn_coffe_predictor\datasets"#Update with the actual path
GLOBAL_START_DATE = "1990-01-01"
GLOBAL_END_DATE = "2023-12-31"
COFFEE_FUTURES_SYMBOL = "KC=F"
GRAPHICS_DIR = "graficos_eda_final"
TAMANHO_IMAGEM = (24,16)
# Ensure graphics directory exists
os.makedirs(GRAPHICS_DIR, exist_ok=True)

def load_datasets(path):
    """Load all relevant CSV files from the specified path into a dictionary of DataFrames."""
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
    """Transform wide-format DataFrames to long format and aggregate by year."""
    master_df = pd.DataFrame()
    for name, df in dataframes.items():
        try:
            # Identify id_vars and year columns
            id_vars = df.columns[0:1].tolist()  # Assuming the first column is the entity/country
            year_cols = df.columns[1:].tolist()  # All other columns are year columns
            
            # Melt the DataFrame
            long_df = df.melt(id_vars=id_vars, var_name='Ano', value_name=f'Valor{name}')
            long_df['Ano'] = long_df['Ano'].astype(int)  # Ensure 'Ano' is integer type
            
            logging.info(f"DataFrame longo para {name} contém {long_df.shape[0]} linhas.")
            
            # Annual aggregation
            aggregated = long_df.groupby('Ano')[f'Valor{name}'].mean().reset_index()
            aggregated.rename(columns={f'Valor{name}': f'{name}_MediaAnual'}, inplace=True)
            
            logging.info(f"DataFrame agregado para {name} contém {aggregated.shape[0]} linhas.")
            
            # Merge into master DataFrame
            if master_df.empty:
                master_df = aggregated
            else:
                master_df = master_df.merge(aggregated, on='Ano', how='outer')
                
            logging.info(f"Transformação e agregação para {name} concluídas.")
        except Exception as e:
            logging.error(f"Erro ao transformar e agregar {name}: {e}")
    
    logging.info(f"Master DataFrame após agregação: {master_df.head()}")
    return master_df

def transform_and_aggregate(dataframes):
    """Transform wide-format DataFrames to long format and aggregate by year."""
    master_df = pd.DataFrame()
    for name, df in dataframes.items():
        try:
            # Identify id_vars and year columns
            id_vars = df.columns[0:1].tolist()  # Assuming the first column is the entity/country
            year_cols = df.columns[1:].tolist()  # All other columns are year columns
            
            # Melt the DataFrame
            long_df = df.melt(id_vars=id_vars, var_name='Ano', value_name=f'Valor{name}')
            long_df['Ano'] = long_df['Ano'].astype(int)  # Ensure 'Ano' is integer type
            
            # Annual aggregation
            aggregated = long_df.groupby('Ano')[f'Valor{name}'].mean().reset_index()
            aggregated.rename(columns={f'Valor{name}': f'{name}_MediaAnual'}, inplace=True)
            
            # Merge into master DataFrame
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
    """Fetch daily coffee futures prices from Yahoo Finance."""
    try:
        coffee_data = yf.download(COFFEE_FUTURES_SYMBOL, start=start_date, end=end_date)
        coffee_data['Year'] = coffee_data.index.year
        coffee_data['Daily_Diff'] = coffee_data['Close'].diff()
        
        # Calculate annual growth trend
        trend = coffee_data.groupby('Year')['Daily_Diff'].sum().reset_index()
        trend.rename(columns={'Daily_Diff': 'TendenciaCrescimentoCafe'}, inplace=True)
        
        logging.info("Dados de preços de café obtidos com sucesso.")
        return trend
    except Exception as e:
        logging.error(f"Erro ao obter dados de preços de café: {e}")
        return pd.DataFrame()

def merge_data(master_df, trend_df):
    """Merge the master DataFrame with the coffee price trend DataFrame."""
    merged_df = master_df.merge(trend_df, left_on='Ano', right_on='Year', how='outer')
    merged_df.drop(columns='Year', inplace=True)
    logging.info("Dados mesclados com sucesso.")
    return merged_df

def perform_eda(df):
    """Perform exploratory data analysis and generate visualizations."""
    # Descriptive statistics
    logging.info("Gerando estatísticas descritivas.")
    stats = df.describe()
    print(stats)

    # Correlation analysis
    logging.info("Calculando a matriz de correlação.")
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(TAMANHO_IMAGEM))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Matriz de Correlação')
    plt.savefig(os.path.join(GRAPHICS_DIR, 'matriz_correlacao.png'))
    plt.close()

    # Time-series visualizations
    for column in df.columns[1:]:
        plt.figure(figsize=(TAMANHO_IMAGEM))
        plt.plot(df['Ano'], df[column], marker='o')
        plt.title(f'Evolução de {column}')
        plt.xlabel('Ano')
        plt.ylabel(column)
        plt.grid()
        plt.savefig(os.path.join(GRAPHICS_DIR, f'evolucao_{column}.png'))
        plt.close()

    # Overlay plot
    plt.figure(figsize=(TAMANHO_IMAGEM))
    plt.plot(df['Ano'], df['TendenciaCrescimentoCafe'], label='Tendência de Crescimento', color='orange')
    plt.plot(df['Ano'], df['disappearance_MediaAnual'], label='Desaparecimento Média Anual', color='blue')
    plt.title('Tendência de Crescimento e Desaparecimento')
    plt.xlabel('Ano')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(GRAPHICS_DIR, 'tendencia_crescimento_overlay.png'))
    plt.close()

    # Distribution analysis
    for column in df.columns[1:]:
        plt.figure(figsize=(TAMANHO_IMAGEM))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribuição de {column}')
        plt.xlabel(column)
        plt.ylabel('Frequência')
        plt.savefig(os.path.join(GRAPHICS_DIR, f'distribuicao_{column}.png'))
        plt.close()

    # Scatter plots
    for column in df.columns[1:]:
        plt.figure(figsize=(TAMANHO_IMAGEM))
        sns.scatterplot(x=df[column], y=df['TendenciaCrescimentoCafe'])
        sns.regplot(x=df[column], y=df['TendenciaCrescimentoCafe'], scatter=False, color='red')
        plt.title(f'Relação entre {column} e Tendência de Crescimento')
        plt.xlabel(column)
        plt.ylabel('Tendência de Crescimento')
        plt.savefig(os.path.join(GRAPHICS_DIR, f'relacao_{column}.png'))
        plt.close()

    # Pair plot
    selected_columns = ['TendenciaCrescimentoCafe'] + df.columns[1:5].tolist()  # Top 4 metrics
    sns.pairplot(df[selected_columns])
    plt.savefig(os.path.join(GRAPHICS_DIR, 'pairplot.png'))
    plt.close()

def main():
    """Main function to execute the data analysis pipeline."""
    logging.info("Iniciando o processo de análise de dados de café.")
    
    # Step 1: Data Acquisition
    dataframes = load_datasets(KAGGLE_DATASET_PATH)
    
    # Step 2: Data Transformation & Aggregation
    master_df = transform_and_aggregate(dataframes)
    
    # Step 3: Coffee Price Growth Trend
    trend_df = fetch_coffee_prices(GLOBAL_START_DATE, GLOBAL_END_DATE)
    
    # Step 4: Merge Data
    final_df = merge_data(master_df, trend_df)
    
    # Step 5: Exploratory Data Analysis
    perform_eda(final_df)

    logging.info("Análise de dados de café concluída com sucesso.")

if __name__ == "__main__":
    main()
