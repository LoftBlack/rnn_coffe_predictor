import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

# ========== Configuração do Logging ==========
os.makedirs("logs", exist_ok=True)
log_file_path = "logs/modelo_previsao.log"

logging.basicConfig(level=logging.INFO, # Mude para logging.DEBUG para ver mais detalhes
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])

logging.info("Iniciando execução do script de previsão de commodities.")

# ========== 1. DOWNLOAD DOS DADOS ==========
logging.info("Iniciando download dos dados históricos de commodities.")
start_date = "2010-01-01"
end_date = "2024-12-31"

try:
    df_cafe = yf.download('KC=F', start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Cafe'})
    df_petroleo = yf.download('CL=F', start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Petroleo'})
    df_milho = yf.download('ZC=F', start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Milho'})
    logging.info("Download de dados concluído com sucesso.")
except Exception as e:
    logging.error(f"Erro ao baixar dados do Yahoo Finance: {e}. Verifique sua conexão com a internet ou os tickers.")
    exit()

# ========== 2. UNIFICAÇÃO E FEATURES ADICIONAIS ==========
logging.info("Unificando DataFrames e calculando features adicionais.")
df = df_cafe.join([df_petroleo, df_milho], how='inner')

# Calcula as taxas de variação diária para todas as commodities
df['Var_Cafe'] = df['Cafe'].pct_change()
df['Var_Petroleo'] = df['Petroleo'].pct_change()
df['Var_Milho'] = df['Milho'].pct_change()

# Remove linhas com valores NaN que surgem após pct_change ou junção
initial_rows = len(df)
df.dropna(inplace=True)
logging.info(f"Removidas {initial_rows - len(df)} linhas com valores NaN.")
logging.info(f"DataFrame final com {len(df)} linhas e {len(df.columns)} colunas.")

# --- CORREÇÃO: GARANTIR NOMES DE COLUNAS COMO STRINGS SIMPLES ---
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
logging.info(f"Nomes das colunas ajustados para: {df.columns.tolist()}")
# --- FIM DA CORREÇÃO ---

# --- VERIFICAÇÕES CRÍTICAS ---
if df.empty:
    logging.error("O DataFrame está vazio após a unificação e remoção de NaN. Não há dados para processar.")
    exit()

if 'Cafe' not in df.columns:
    logging.error("A coluna 'Cafe' não foi encontrada no DataFrame. Verifique os nomes das colunas e o processo de download.")
    logging.debug(f"Colunas presentes no DataFrame: {df.columns.tolist()}")
    exit()
# --- FIM DAS VERIFICAÇÕES CRÍTICAS ---

# ========== 3. NORMALIZAÇÃO ==========
logging.info("Iniciando normalização dos dados (MinMaxScaler).")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

feature_names = df.columns.tolist()
logging.info(f"Colunas usadas para treinamento (features) após ajuste: {feature_names}")
logging.debug(f"Primeiras 5 linhas do scaled_data:\n{scaled_data[:5]}")
logging.info("Normalização concluída.")

# ========== 4. CRIAÇÃO DAS JANELAS (5 DIAS DE HISTÓRICO -> 5 DIAS DE PREVISÃO) ==========
logging.info("Criando janelas de dados para o modelo LSTM.")
window_size = 5
forecast_horizon = 5

X, y = [], []
cafe_feature_index = feature_names.index('Cafe')

for i in range(len(scaled_data) - window_size - forecast_horizon):
    X.append(scaled_data[i:i+window_size])
    y.append(scaled_data[i+window_size:i+window_size+forecast_horizon, cafe_feature_index])

X = np.array(X)
y = np.array(y)
logging.info(f"Janelas criadas. X shape: {X.shape}, y shape: {y.shape}")

if X.shape[0] == 0 or y.shape[0] == 0:
    logging.error("Não há amostras suficientes para criar janelas após a filtragem. Verifique o tamanho dos dados e os parâmetros window_size/forecast_horizon.")
    exit()

# ========== 5. DIVISÃO TREINO/VALIDAÇÃO/TESTE ==========
logging.info("Dividindo os dados em conjuntos de treino, validação e teste (60/20/20).")
total_samples = len(X)
train_split = int(0.6 * total_samples)
val_split = int(0.8 * total_samples)

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

logging.info(f"Total de amostras: {total_samples}")
logging.info(f"Amostras de treino: {len(X_train)}")
logging.info(f"Amostras de validação: {len(X_val)}")
logging.info(f"Amostras de teste: {len(X_test)}")

# ========== 6. MODELO LSTM ==========
logging.info("Construindo o modelo LSTM.")
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(window_size, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(forecast_horizon))

model.compile(optimizer='adam', loss='mse')
model.summary()
logging.info("Modelo LSTM compilado.")

# ========== 7. TREINAMENTO ==========
logging.info("Iniciando treinamento do modelo LSTM (50 épocas).")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)
logging.info("Treinamento do modelo concluído.")

# ========== 8. PREVISÃO ==========
logging.info("Realizando previsões no conjunto de teste.")
predicted = model.predict(X_test)
logging.info("Previsões concluídas.")

# ========== 9. DESNORMALIZAÇÃO ==========
logging.info("Desnormalizando os resultados das previsões (apenas preço do Café).")
cafe_index = feature_names.index('Cafe')

scaler_cafe_only = MinMaxScaler()
scaler_cafe_only.min_ = scaler.min_[cafe_index:cafe_index+1]
scaler_cafe_only.scale_ = scaler.scale_[cafe_index:cafe_index+1]
scaler_cafe_only.data_min_ = scaler.data_min_[cafe_index:cafe_index+1]
scaler_cafe_only.data_max_ = scaler.data_max_[cafe_index:cafe_index+1]
scaler_cafe_only.data_range_ = scaler.data_range_[cafe_index:cafe_index+1]

y_test_rescaled = scaler_cafe_only.inverse_transform(y_test)
predicted_rescaled = scaler_cafe_only.inverse_transform(predicted)
logging.info("Desnormalização concluída.")

# ========== 10. ORGANIZAÇÃO DAS SAÍDAS (para métricas e gráficos específicos) ==========
y_test_day1 = y_test_rescaled[:, 0]
y_pred_day1 = predicted_rescaled[:, 0]
y_test_day5 = y_test_rescaled[:, 4]
y_pred_day5 = predicted_rescaled[:, 4]

# ========== 11. CRIAÇÃO DO DIRETÓRIO DE GRÁFICOS ==========
logging.info("Criando diretório para salvar gráficos.")
os.makedirs("graficos", exist_ok=True)

def salvar_grafico(nome):
    plt.tight_layout()
    plt.savefig(f'graficos/{nome}')
    plt.close()

# ========== 12. GRÁFICO DE DISPERSÃO: DIA +1 ==========
logging.info("Gerando gráfico de dispersão para o Dia +1.")
plt.figure(figsize=(8, 6))
plt.scatter(y_test_day1, y_pred_day1, alpha=0.5, color='blue')
plt.plot([min(y_test_day1), max(y_test_day1)],
         [min(y_test_day1), max(y_test_day1)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +1 Real vs Previsto (Café)')
plt.xlabel('Real')
plt.ylabel('Previsto')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_1.png')

# ========== 13. GRÁFICO DE DISPERSÃO: DIA +5 ==========
logging.info("Gerando gráfico de dispersão para o Dia +5.")
plt.figure(figsize=(8, 6))
plt.scatter(y_test_day5, y_pred_day5, alpha=0.5, color='green')
plt.plot([min(y_test_day5), max(y_test_day5)],
         [min(y_test_day5), max(y_test_day5)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +5 Real vs Previsto (Café)')
plt.xlabel('Real')
plt.ylabel('Previsto')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_5.png')

# ========== 14. HISTÓRICO DE PERDA ==========
logging.info("Gerando gráfico do histórico de perda (Loss).")
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Histórico de Perda (Loss)')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
salvar_grafico('historico_perda.png')

# ========== 15. NOVO GRÁFICO: COMPARAÇÃO TEMPORAL PARA DIA +1 E DIA +5 APENAS ==========
logging.info("Gerando gráfico de comparação temporal para Dia +1 e Dia +5.")
plt.figure(figsize=(15, 7))

test_sample_indices = np.arange(len(y_test_rescaled))

# Plot para Dia +1
plt.plot(test_sample_indices, y_test_rescaled[:, 0], label='Real (Dia +1)', linestyle='-', color='blue')
plt.plot(test_sample_indices, predicted_rescaled[:, 0], label='Previsto (Dia +1)', linestyle='--', color='red')

# Plot para Dia +5
plt.plot(test_sample_indices, y_test_rescaled[:, 4], label='Real (Dia +5)', linestyle='-', color='green')
plt.plot(test_sample_indices, predicted_rescaled[:, 4], label='Previsto (Dia +5)', linestyle='--', color='orange')


plt.title('Comparação Temporal - Previsões para Dia +1 e Dia +5 (Café)')
plt.xlabel('Amostra de Teste')
plt.ylabel('Preço do Café')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
salvar_grafico('comparacao_temporal_dia_1_e_5.png') # Nome do arquivo alterado

# ========== 16. NOVO GRÁFICO: DISPERSÃO PARA DIA +1 E DIA +5 APENAS ==========
logging.info("Gerando gráfico de dispersão para Dia +1 e Dia +5 combinados.")
plt.figure(figsize=(10, 8))

# Concatena os dados do Dia +1 e Dia +5
combined_y_test = np.concatenate((y_test_day1, y_test_day5))
combined_y_pred = np.concatenate((y_pred_day1, y_pred_day5))

plt.scatter(combined_y_test, combined_y_pred, alpha=0.5, color='purple')
min_val = min(combined_y_test.min(), combined_y_pred.min())
max_val = max(combined_y_test.max(), combined_y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
plt.title('Dispersão: Previsões Dia +1 e Dia +5 (Real vs Previsto - Café)')
plt.xlabel('Real')
plt.ylabel('Previsto')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_1_e_5.png') # Nome do arquivo alterado

# ========== 17. MÉTRICAS ==========
logging.info("Calculando e exibindo métricas de desempenho.")
print("\n========== MÉTRICAS DE DESEMPENHO (Preço do Café) ==========")

print("\n--- Dia +1 ---")
r2_day1 = r2_score(y_test_day1, y_pred_day1)
mae_day1 = mean_absolute_error(y_test_day1, y_pred_day1)
mse_day1 = mean_squared_error(y_test_day1, y_pred_day1)
print(f'R²:  {r2_day1:.4f}')
print(f'MAE: ${mae_day1:.4f}')
print(f'MSE: ${mse_day1:.4f}')
logging.info(f"Métricas Dia +1: R²={r2_day1:.4f}, MAE=${mae_day1:.4f}, MSE=${mse_day1:.4f}")

print("\n--- Dia +5 ---")
r2_day5 = r2_score(y_test_day5, y_pred_day5)
mae_day5 = mean_absolute_error(y_test_day5, y_pred_day5)
mse_day5 = mean_squared_error(y_test_day5, y_pred_day5)
print(f'R²:  {r2_day5:.4f}')
print(f'MAE: ${mae_day5:.4f}')
print(f'MSE: ${mse_day5:.4f}')
logging.info(f"Métricas Dia +5: R²={r2_day5:.4f}, MAE=${mae_day5:.4f}, MSE=${mse_day5:.4f}")

# Metrics for all 5 days (average) - Isso aqui não é mais "todos os 5 dias", mas sim uma média do dia 1 e 5 combinados para a dispersão
# Renomeando para refletir que a métrica é para o conjunto combinado de Dia +1 e Dia +5
print("\n--- Média para Dia +1 e Dia +5 combinados ---")
r2_combined_days = r2_score(combined_y_test, combined_y_pred)
mae_combined_days = mean_absolute_error(combined_y_test, combined_y_pred)
mse_combined_days = mean_squared_error(combined_y_test, combined_y_pred)
print(f'R² (Combinado):  {r2_combined_days:.4f}')
print(f'MAE (Combinado): ${mae_combined_days:.4f}')
print(f'MSE (Combinado): ${mse_combined_days:.4f}')
logging.info(f"Métricas Combinadas Dia +1 e Dia +5: R²={r2_combined_days:.4f}, MAE=${mae_combined_days:.4f}, MSE=${mse_combined_days:.4f}")

logging.info("Execução do script concluída.")