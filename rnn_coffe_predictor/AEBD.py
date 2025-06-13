import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# =========================
# 1. COLETA E PRÉ-PROCESSAMENTO
# =========================
def baixar_e_processar_cafe(start='2018-01-01', end='2024-12-31'):
    print("[INFO] Baixando dados do café (KC=F) via yfinance...")
    df = yf.download('KC=F', start=start, end=end)

    if df.empty:
        raise ValueError("Erro: Dados do café não foram baixados corretamente.")

    # Seleciona apenas o fechamento e remove MultiIndex das colunas, se houver
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

    df = df[['Close_KC=F']]  # Nome da coluna após flatten
    df.rename(columns={'Close_KC=F': 'PrecoCafe'}, inplace=True)

    df.index.name = 'Date'
    print("[OK] Dados do café processados com colunas:", df.columns.tolist())
    return df

def baixar_e_processar_clima():
    print("[INFO] Baixando dados climáticos via kagglehub...")
    path = kagglehub.dataset_download("gnomows/dados-metereologicos-2018-2024-inmet")
    print("[OK] Arquivos baixados em:", path)

    arquivos = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not arquivos:
        raise FileNotFoundError("Erro: Nenhum arquivo CSV encontrado na pasta do dataset KaggleHub.")

    caminho_csv = os.path.join(path, arquivos[0])
    print("[INFO] Lendo arquivo climático:", caminho_csv)
    df = pd.read_csv(caminho_csv, low_memory=False)

    df['DateTime'] = pd.to_datetime(df['Data'] + ' ' + df['HORA (UTC)'], errors='coerce')
    df = df.set_index('DateTime').drop(columns=['Data', 'HORA (UTC)'])

    column_mapping = {
        'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'TemperaturaMediaC',
        'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'TemperaturaMaximaDiariaC',
        'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'TemperaturaMinimaDiariaC',
        'UMIDADE RELATIVA DO AR, HORARIA (%)': 'UmidadeRelativaAr',
        'VENTO, VELOCIDADE HORARIA (m/s)': 'VelocidadeVento',
        'RADIACAO GLOBAL (KJ/m²)': 'RadiacaoGlobal',
        'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'PrecipitacaoTotal'
    }

    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if col != 'PrecipitacaoTotal':
            df[col] = df[col].replace(-9999, np.nan)
            df[col] = df[col].mask((df[col] < -80) | (df[col] > 60), np.nan)
        else:
            df[col] = df[col].replace(-9999, np.nan)

    df = df.interpolate(method='linear', limit_direction='both')
    df = df.dropna()

    agregacoes = {
        'PrecipitacaoTotal': 'sum',
        'RadiacaoGlobal': 'sum',
        'TemperaturaMediaC': 'mean',
        'UmidadeRelativaAr': 'mean',
        'VelocidadeVento': 'mean',
        'TemperaturaMaximaDiariaC': 'mean',
        'TemperaturaMinimaDiariaC': 'mean'
    }

    df_diario = df.resample('D').agg({k: v for k, v in agregacoes.items() if k in df.columns})
    df_diario = df_diario.dropna()

    print("[OK] Dados climáticos processados com colunas:", df_diario.columns.tolist())
    return df_diario

# =========================
# Função para normalizar data no df café
# =========================
def normaliza_data_cafe(df_cafe):
    """
    Remove a hora do índice datetime do dataframe do café,
    deixando só a data para garantir merge correto.
    """
    df = df_cafe.copy()
    df.index = pd.to_datetime(df.index).normalize()
    return df

# =========================
# 2. UNIÃO DOS DADOS
# =========================
def unir_dados(df_cafe, df_clima):
    print("[INFO] Normalizando índice do café (remoção de horas)...")
    df_cafe_norm = normaliza_data_cafe(df_cafe)

    print("[INFO] Unindo os dados de café e clima...")
    df_cafe_ = df_cafe_norm.reset_index()
    df_clima_ = df_clima.reset_index()

    # Como clima já está diário (index é só data), garantimos também no reset_index
    df_clima_['Date'] = pd.to_datetime(df_clima_['Date']).dt.normalize()
    df_cafe_['Date'] = pd.to_datetime(df_cafe_['Date']).dt.normalize()

    print("[DEBUG] df_cafe_ columns:", df_cafe_.columns)
    print("[DEBUG] df_clima_ columns:", df_clima_.columns)

    df_completo = pd.merge(df_cafe_, df_clima_, on='Date', how='inner')
    df_completo.set_index('Date', inplace=True)

    print("[OK] Dados unidos com shape:", df_completo.shape)
    return df_completo

# =========================
# 3. ANÁLISE EXPLORATÓRIA
# =========================
def realizar_eda(df_final):
    os.makedirs('graficos_eda', exist_ok=True)

    print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
    print(df_final.describe())

    df_final['PrecoCafe'].plot(title='Preço do Café (2018-2024)', figsize=(12, 4))
    plt.ylabel('US$/lb')
    plt.tight_layout()
    plt.savefig('graficos_eda/serie_preco_cafe.png')
    plt.close()

    for coluna in df_final.columns:
        if coluna != 'PrecoCafe':
            df_final[coluna].plot(title=f'{coluna} (2018-2024)', figsize=(12, 4))
            plt.tight_layout()
            plt.savefig(f'graficos_eda/serie_{coluna}.png')
            plt.close()

    df_final.hist(bins=30, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('graficos_eda/histogramas.png')
    plt.close()

    df_final.plot(kind='box', subplots=True, layout=(3, 3), figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('graficos_eda/boxplots.png')
    plt.close()

    corr = df_final.corr()
    print("\n=== MATRIZ DE CORRELAÇÃO ===")
    print(corr)

    corrs_com_cafe = corr['PrecoCafe'].drop('PrecoCafe').sort_values(key=lambda x: abs(x), ascending=False)
    print("\n=== CORRELAÇÕES COM O PREÇO DO CAFÉ ===")
    print(corrs_com_cafe)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('graficos_eda/heatmap_correlacoes.png')
    plt.close()

# =========================
# MAIN SCRIPT
# =========================
if __name__ == '__main__':

    df_cafe = baixar_e_processar_cafe()
    df_clima = baixar_e_processar_clima()

    print("[INFO] Primeiras datas do café:", df_cafe.index.min(), "->", df_cafe.index.max())
    print("[INFO] Primeiras datas do clima:", df_clima.index.min(), "->", df_clima.index.max())

    df_final = unir_dados(df_cafe, df_clima)

    print("[INFO] Colunas do DataFrame Final:", df_final.columns.tolist())
    print("[INFO] Primeiras linhas do DataFrame Final:")
    print(df_final.head())

    realizar_eda(df_final)
