import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ========== CONFIGURAÇÃO DO LOGGING ==========
os.makedirs("logs", exist_ok=True)
log_file_path = "logs/modelo_janela5_entradas_triplas_precos.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def log_section(title):
    logging.info("=" * 60)
    logging.info(f" >>> {title.upper()} <<< ")
    logging.info("=" * 60)

log_section("Iniciando Execução do Modelo LSTM (Preços Brutos)")

# ========== 1. DOWNLOAD DOS DADOS ==========
log_section("1. Download dos Dados Históricos")
start_date = "2010-01-01"
end_date = "2024-12-31"

try:
    logging.info(f"Baixando dados do Yahoo Finance ({start_date} até {end_date})...")
    df_cafe = yf.download('KC=F', start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Cafe'})
    df_petroleo = yf.download('CL=F', start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Petroleo'})
    df_milho = yf.download('ZC=F', start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Milho'})
    logging.info("Download de dados concluído com sucesso.")
except Exception as e:
    logging.error(f"Erro ao baixar dados do Yahoo Finance: {e}")
    exit()

# ========== 2. UNIFICAÇÃO E LIMPEZA DOS DADOS ==========
log_section("2. Unificação e Tratamento dos Dados")
df = df_cafe.join([df_petroleo, df_milho], how='inner')

# Ajusta nomes de colunas caso haja MultiIndex vindo do yfinance
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# Remove linhas com dados ausentes (sem aplicar pct_change)
initial_rows = len(df)
df.dropna(inplace=True)
logging.info(f"Linhas iniciais: {initial_rows} | Linhas após limpeza: {len(df)} (Removidas: {initial_rows - len(df)})")
logging.info(f"Colunas ativas no dataset: {df.columns.tolist()}")

# ========== 3. PREPARAÇÃO E NORMALIZAÇÃO ==========
log_section("3. Normalização dos Dados")
feature_names = ['Cafe', 'Petroleo', 'Milho']
target_name = 'Cafe'

logging.info(f"Features de entrada: {feature_names}")
logging.info(f"Target de previsão: {target_name}")

scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

scaled_features = scaler_features.fit_transform(df[feature_names].values)
scaled_target = scaler_target.fit_transform(df[[target_name]].values)

logging.info(f"Shape das Features Normalizadas: {scaled_features.shape}")
logging.info(f"Shape do Target Normalizado: {scaled_target.shape}")

# ========== 4. CRIAÇÃO DAS JANELAS TEMPORAIS ==========
log_section("4. Construção das Janelas (5 Dias Entrada -> 5 Dias Previsão)")
window_size = 5
forecast_horizon = 5

X, y = [], []
for i in range(len(scaled_features) - window_size - forecast_horizon + 1):
    X.append(scaled_features[i : i + window_size])
    y.append(scaled_target[i + window_size : i + window_size + forecast_horizon, 0])

X, y = np.array(X), np.array(y)
logging.info(f"Janelas construídas com sucesso | X shape: {X.shape} | y shape: {y.shape}")

if len(X) == 0:
    logging.error("Conjunto de dados insuficiente para criar as janelas.")
    exit()

# ========== 5. DIVISÃO TREINO / VALIDAÇÃO / TESTE ==========
log_section("5. Divisão dos Dados (60% Treino / 20% Validação / 20% Teste)")
total_samples = len(X)
train_split = int(0.6 * total_samples)
val_split = int(0.8 * total_samples)

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

logging.info(f"Amostras Totais: {total_samples}")
logging.info(f"Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

# ========== 6. ARQUITETURA DO MODELO LSTM PRINCIPAL ==========
log_section("6. Compilação e Treinamento do Modelo Principal (3 Entradas)")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(window_size, X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dense(forecast_horizon)
])

model.compile(optimizer='adam', loss='mse')
logging.info("Arquitetura da rede neural compilada:")
model.summary(print_fn=lambda x: logging.info(x))

logging.info("Iniciando treinamento (50 Épocas)...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)
logging.info("Treinamento do modelo principal concluído.")

# ========== 7. PREVISÃO E DESNORMALIZAÇÃO ==========
log_section("7. Avaliação e Desnormalização de Preços")
predicted_scaled = model.predict(X_test)

# Desnormalização direta para preços reais em USD
y_price_true = scaler_target.inverse_transform(y_test)
y_price_pred = scaler_target.inverse_transform(predicted_scaled)

y_price_test_day1, y_price_pred_day1 = y_price_true[:, 0], y_price_pred[:, 0]
y_price_test_day5, y_price_pred_day5 = y_price_true[:, 4], y_price_pred[:, 4]

combined_y_test = np.concatenate((y_price_test_day1, y_price_test_day5))
combined_y_pred = np.concatenate((y_price_pred_day1, y_price_pred_day5))

# ========== 8. CRIAÇÃO E SALVAMENTO DOS GRÁFICOS ==========
log_section("8. Geração de Gráficos")
os.makedirs("graficos", exist_ok=True)

def salvar_grafico(nome):
    plt.tight_layout()
    plt.savefig(f'graficos/{nome}')
    plt.close()

# Dispersão Dia +1
plt.figure(figsize=(8, 6))
plt.scatter(y_price_test_day1, y_price_pred_day1, alpha=0.5, color='blue')
plt.plot([min(y_price_test_day1), max(y_price_test_day1)], [min(y_price_test_day1), max(y_price_test_day1)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +1 Real vs Previsto (Café) - Preços Brutos')
plt.xlabel('Real (USD)')
plt.ylabel('Previsto (USD)')
plt.legend(); plt.grid(True)
salvar_grafico('dispersao_dia_1_3entradas_precos.png')

# Dispersão Dia +5
plt.figure(figsize=(8, 6))
plt.scatter(y_price_test_day5, y_price_pred_day5, alpha=0.5, color='green')
plt.plot([min(y_price_test_day5), max(y_price_test_day5)], [min(y_price_test_day5), max(y_price_test_day5)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +5 Real vs Previsto (Café) - Preços Brutos')
plt.xlabel('Real (USD)')
plt.ylabel('Previsto (USD)')
plt.legend(); plt.grid(True)
salvar_grafico('dispersao_dia_5_3entradas_precos.png')

# Histórico de Perda (Loss)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Histórico de Perda (Loss) - Preços Brutos')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend(); plt.grid(True)
salvar_grafico('historico_perda_3entradas_precos.png')

# Comparação Temporal
plt.figure(figsize=(15, 7))
test_sample_indices = np.arange(len(y_price_true))
plt.plot(test_sample_indices, y_price_true[:, 0], label='Real (Dia +1)', color='blue')
plt.plot(test_sample_indices, y_price_pred[:, 0], label='Previsto (Dia +1)', linestyle='--', color='red')
plt.plot(test_sample_indices, y_price_true[:, 4], label='Real (Dia +5)', color='green')
plt.plot(test_sample_indices, y_price_pred[:, 4], label='Previsto (Dia +5)', linestyle='--', color='orange')
plt.title('Comparação Temporal (Café) - Preços Brutos')
plt.xlabel('Amostra de Teste')
plt.ylabel('Preço do Café (USD)')
plt.legend(); plt.grid(True)
salvar_grafico('comparacao_temporal_3entradas_precos.png')

logging.info("Gráficos gerados e salvos com sucesso no diretório 'graficos/'.")

# ========== 9. CÁLCULO E LOGGING DAS MÉTRICAS PRINCIPAIS ==========
log_section("9. Resultados do Modelo Principal (3 Entradas)")

def exibir_metricas(tag, true_vals, pred_vals):
    r2 = r2_score(true_vals, pred_vals)
    mae = mean_absolute_error(true_vals, pred_vals)
    mse = mean_squared_error(true_vals, pred_vals)
    msg = f"[{tag}] R²: {r2:.4f} | MAE: ${mae:.4f} | MSE: ${mse:.4f}"
    print(msg)
    logging.info(msg)
    return r2, mae, mse

exibir_metricas("Dia +1", y_price_test_day1, y_price_pred_day1)
exibir_metricas("Dia +5", y_price_test_day5, y_price_pred_day5)
exibir_metricas("Combinado (Dia +1 e +5)", combined_y_test, combined_y_pred)

# ========== CONTROLE 1: BASELINE INGÊNUO (ÚLTIMO PREÇO CONHECIDO) ==========
log_section("10. Controle 1: Baseline Ingênuo")

# Pega o último preço bruto de café no final de cada janela de teste
last_known_prices = df['Cafe'].iloc[val_split + window_size - 1 : val_split + window_size - 1 + len(X_test)].values
y_price_baseline = np.tile(last_known_prices.reshape(-1, 1), (1, forecast_horizon))

y_price_baseline_day1 = y_price_baseline[:, 0]
y_price_baseline_day5 = y_price_baseline[:, 4]
combined_y_baseline = np.concatenate((y_price_baseline_day1, y_price_baseline_day5))

exibir_metricas("Baseline - Dia +1", y_price_test_day1, y_price_baseline_day1)
exibir_metricas("Baseline - Dia +5", y_price_test_day5, y_price_baseline_day5)
exibir_metricas("Baseline - Combinado", combined_y_test, combined_y_baseline)

# ========== CONTROLE 2: LSTM ENTRADA ÚNICA (APENAS CAFÉ) ==========
log_section("11. Controle 2: LSTM Univariado (Apenas Café)")

features_c2 = df[['Cafe']].values
scaled_features_c2 = scaler_target.transform(features_c2)

X_c2, y_c2 = [], []
for i in range(len(scaled_features_c2) - window_size - forecast_horizon + 1):
    X_c2.append(scaled_features_c2[i : i + window_size])
    y_c2.append(scaled_target[i + window_size : i + window_size + forecast_horizon, 0])

X_c2, y_c2 = np.array(X_c2), np.array(y_c2)
X_train_c2, y_train_c2 = X_c2[:train_split], y_c2[:train_split]
X_val_c2, y_val_c2 = X_c2[train_split:val_split], y_c2[train_split:val_split]
X_test_c2, y_test_c2 = X_c2[val_split:], y_c2[val_split:]

model_c2 = Sequential([
    LSTM(128, return_sequences=True, input_shape=(window_size, X_c2.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dense(forecast_horizon)
])
model_c2.compile(optimizer='adam', loss='mse')

logging.info("Treinando Modelo Controle 2 (Entrada Única)...")
model_c2.fit(X_train_c2, y_train_c2, epochs=50, batch_size=32, validation_data=(X_val_c2, y_val_c2), verbose=0)

predicted_scaled_c2 = model_c2.predict(X_test_c2)
y_price_pred_c2 = scaler_target.inverse_transform(predicted_scaled_c2)

exibir_metricas("Controle 2 - Dia +1", y_price_test_day1, y_price_pred_c2[:, 0])
exibir_metricas("Controle 2 - Dia +5", y_price_test_day5, y_price_pred_c2[:, 4])
exibir_metricas("Controle 2 - Combinado", combined_y_test, np.concatenate((y_price_pred_c2[:, 0], y_price_pred_c2[:, 4])))

log_section("Execução Finalizada com Sucesso")