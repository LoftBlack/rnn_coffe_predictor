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
log_file_path = "logs/modelo_janela5_entradas_triplas.log"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])

logging.info("Iniciando execução do modelo LSTM com janela de 5 dias e 3 entradas (café, petróleo, milho).")

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

# ========== 2. UNIFICAÇÃO E CÁLCULO DE FEATURES ==========
logging.info("Unificando DataFrames e calculando features adicionais.")
df = df_cafe.join([df_petroleo, df_milho], how='inner')

# Calcula as taxas de variação diária para todas as commodities (seguindo o padrão do projeto)
df['Var_Cafe'] = df['Cafe'].pct_change()
df['Var_Petroleo'] = df['Petroleo'].pct_change()
df['Var_Milho'] = df['Milho'].pct_change()

# Remove linhas com valores NaN que surgem após pct_change ou junção
initial_rows = len(df)
df.dropna(inplace=True)
logging.info(f"Removidas {initial_rows - len(df)} linhas com valores NaN.")
logging.info(f"DataFrame final com {len(df)} linhas e {len(df.columns)} colunas.")

# Garantir nomes de colunas como strings simples
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
logging.info(f"Nomes das colunas ajustados para: {df.columns.tolist()}")

# ========== 3. PREPARAÇÃO DOS DADOS PARA MODELAGEM ==========
# Usaremos as taxas de variação (pct_change) como features de entrada
# e prevemos as taxas de variação futura do café (que serão convertidas para preços)
feature_names = ['Var_Cafe', 'Var_Petroleo', 'Var_Milho']
target_name = 'Var_Cafe'  # Vamos prever a variação futura do café

logging.info(f"Features de entrada: {feature_names}")
logging.info(f"Target: {target_name}")

# Extrair os dados
features_data = df[feature_names].values
target_data = df[target_name].values

# ========== 4. NORMALIZAÇÃO ==========
logging.info("Iniciando normalização dos dados (MinMaxScaler).")
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_features = scaler_features.fit_transform(features_data)
scaled_target = scaler_target.fit_transform(target_data.reshape(-1, 1)).flatten()

logging.info(f"Features normalizadas. Shape: {scaled_features.shape}")
logging.info(f"Target normalizado. Shape: {scaled_target.shape}")

# ========== 5. CRIAÇÃO DAS JANELAS (5 DIAS DE HISTÓRICO -> 5 DIAS DE PREVISÃO) ==========
logging.info("Criando janelas de dados para o modelo LSTM.")
window_size = 5
forecast_horizon = 5

X, y = [], []

for i in range(len(scaled_features) - window_size - forecast_horizon + 1):
    X.append(scaled_features[i:i+window_size])  # window_size passos de 3 features
    y.append(scaled_target[i+window_size:i+window_size+forecast_horizon])  # próximos forecast_horizon passos do target

X = np.array(X)
y = np.array(y)
logging.info(f"Janelas criadas. X shape: {X.shape}, y shape: {y.shape}")

if X.shape[0] == 0 or y.shape[0] == 0:
    logging.error("Não há amostras suficientes para criar janelas após a filtragem. Verifique o tamanho dos dados e os parâmetros window_size/forecast_horizon.")
    exit()

# ========== 6. DIVISÃO TREINO/VALIDAÇÃO/TESTE ==========
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

# ========== 7. MODELO LSTM (ARCHITETURA CONFORME DOCUMENTADA) ==========
logging.info("Construindo o modelo LSTM conforme arquitetura documentada.")
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(window_size, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(forecast_horizon))

model.compile(optimizer='adam', loss='mse')
model.summary()
logging.info("Modelo LSTM compilado.")

# ========== 8. TREINAMENTO ==========
logging.info("Iniciando treinamento do modelo LSTM (50 épocas).")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)
logging.info("Treinamento do modelo concluído.")

# ========== 9. PREVISÃO ==========
logging.info("Realizando previsões no conjunto de teste.")
predicted_scaled = model.predict(X_test)
logging.info("Previsões concluídas.")

# ========== 10. DESNORMALIZAÇÃO ==========
logging.info("Desnormalizando os resultados das previsões (taxa de variação do café).")
y_rescaled = scaler_target.inverse_transform(y_test)
predicted_rescaled = scaler_target.inverse_transform(predicted_scaled)
logging.info("Desnormalização concluída.")

# ========== 11. CONVERSÃO DE TAXA DE VARIAÇÃO PARA PREÇOS ==========
# Para converter as previsões de taxa de variação de volta para preços,
# precisamos aplicar as mudanças percentuais ao último preço conhecido
logging.info("Convertendo previsões de taxa de variação para preços.")

# Obter os últimos preços conhecidos de cada amostra de teste
# A amostra i em X_test corresponde ao window que termina no índice (val_split + i + window_size - 1) no dataframe original
# O preço do café nesse ponto é df['Cafe'].iloc[val_split + i + window_size - 1]
last_known_prices = []
for i in range(len(X_test)):
    # Índice no dataframe original do último dia da janela de entrada
    df_idx = val_split + i + window_size - 1
    last_known_price = df['Cafe'].iloc[df_idx]
    last_known_prices.append(last_known_price)

last_known_prices = np.array(last_known_prices)

# Converter previsões de pct_change para preços
# Se temos pct_change de t para t+1, então: price(t+1) = price(t) * (1 + pct_change)
# Para múltiplos passos: price(t+h) = price(t) * product_{k=1 to h} (1 + pct_change(t+k-1))

y_price_pred = np.zeros_like(y_rescaled)
y_price_true = np.zeros_like(y_rescaled)

for i in range(len(X_test)):
    last_price = last_known_prices[i]
    
    # Preço real futuro
    price_true = last_price
    for j in range(forecast_horizon):
        price_true = price_true * (1 + y_rescaled[i, j])
        y_price_true[i, j] = price_true
    
    # Preço previsto futuro
    price_pred = last_price
    for j in range(forecast_horizon):
        price_pred = price_pred * (1 + predicted_rescaled[i, j])
        y_price_pred[i, j] = price_pred

logging.info("Conversão para preços concluída.")

# ========== 12. ORGANIZAÇÃO DAS SAÍDAS (para métricas e gráficos específicos) ==========
y_price_test_day1 = y_price_true[:, 0]  # Dia +1
y_price_pred_day1 = y_price_pred[:, 0]  # Dia +1
y_price_test_day5 = y_price_true[:, 4]  # Dia +5
y_price_pred_day5 = y_price_pred[:, 4]  # Dia +5

# ========== 13. CRIAÇÃO DO DIRETÓRIO DE GRÁFICOS ==========
logging.info("Criando diretório para salvar gráficos.")
os.makedirs("graficos", exist_ok=True)

def salvar_grafico(nome):
    plt.tight_layout()
    plt.savefig(f'graficos/{nome}')
    plt.close()

# ========== 14. GRÁFICO DE DISPERSÃO: DIA +1 ==========
logging.info("Gerando gráfico de dispersão para o Dia +1.")
plt.figure(figsize=(8, 6))
plt.scatter(y_price_test_day1, y_price_pred_day1, alpha=0.5, color='blue')
plt.plot([min(y_price_test_day1), max(y_price_test_day1)],
         [min(y_price_test_day1), max(y_price_test_day1)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +1 Real vs Previsto (Café) - Modelo 3 Entradas')
plt.xlabel('Real (USD)')
plt.ylabel('Previsto (USD)')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_1_3entradas.png')

# ========== 15. GRÁFICO DE DISPERSÃO: DIA +5 ==========
logging.info("Gerando gráfico de dispersão para o Dia +5.")
plt.figure(figsize=(8, 6))
plt.scatter(y_price_test_day5, y_price_pred_day5, alpha=0.5, color='green')
plt.plot([min(y_price_test_day5), max(y_price_test_day5)],
         [min(y_price_test_day5), max(y_price_test_day5)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +5 Real vs Previsto (Café) - Modelo 3 Entradas')
plt.xlabel('Real (USD)')
plt.ylabel('Previsto (USD)')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_5_3entradas.png')

# ========== 16. HISTÓRICO DE PERDA ==========
logging.info("Gerando gráfico do histórico de perda (Loss).")
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Histórico de Perda (Loss) - Modelo 3 Entradas')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
salvar_grafico('historico_perda_3entradas.png')

# ========== 17. GRÁFICO DE COMPARAÇÃO TEMPORAL ==========
logging.info("Gerando gráfico de comparação temporal para Dia +1 e Dia +5.")
plt.figure(figsize=(15, 7))

test_sample_indices = np.arange(len(y_price_true))

# Plot para Dia +1
plt.plot(test_sample_indices, y_price_true[:, 0], label='Real (Dia +1)', linestyle='-', color='blue')
plt.plot(test_sample_indices, y_price_pred[:, 0], label='Previsto (Dia +1)', linestyle='--', color='red')

# Plot para Dia +5
plt.plot(test_sample_indices, y_price_true[:, 4], label='Real (Dia +5)', linestyle='-', color='green')
plt.plot(test_sample_indices, y_price_pred[:, 4], label='Previsto (Dia +5)', linestyle='--', color='orange')

plt.title('Comparação Temporal - Previsões para Dia +1 e Dia +5 (Café) - Modelo 3 Entradas')
plt.xlabel('Amostra de Teste')
plt.ylabel('Preço do Café (USD)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
salvar_grafico('comparacao_temporal_dia_1_e_5_3entradas.png')

# ========== 18. GRÁFICO DE DISPERSÃO COMBINADO ==========
logging.info("Gerando gráfico de dispersão para Dia +1 e Dia +5 combinados.")
plt.figure(figsize=(10, 8))

# Concatena os dados do Dia +1 e Dia +5
combined_y_test = np.concatenate((y_price_test_day1, y_price_test_day5))
combined_y_pred = np.concatenate((y_price_pred_day1, y_price_pred_day5))

plt.scatter(combined_y_test, combined_y_pred, alpha=0.5, color='purple')
min_val = min(combined_y_test.min(), combined_y_pred.min())
max_val = max(combined_y_test.max(), combined_y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
plt.title('Dispersão: Previsões Dia +1 e Dia +5 (Real vs Previsto - Café) - Modelo 3 Entradas')
plt.xlabel('Real (USD)')
plt.ylabel('Previsto (USD)')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_1_e_5_3entradas.png')

# ========== 19. MÉTRICAS DO MODELO PRINCIPAL ==========
logging.info("Calculando e exibindo métricas de desempenho do modelo principal.")
print("\\n========== MÉTRICAS DE DESEMPENHO - MODELO 3 ENTRADAS (Café, Petróleo, Milho) ==========")

print("\\n--- Dia +1 ---")
r2_day1 = r2_score(y_price_test_day1, y_price_pred_day1)
mae_day1 = mean_absolute_error(y_price_test_day1, y_price_pred_day1)
mse_day1 = mean_squared_error(y_price_test_day1, y_price_pred_day1)
print(f'R²:  {r2_day1:.4f}')
print(f'MAE: ${mae_day1:.4f}')
print(f'MSE: ${mse_day1:.4f}')
logging.info(f"Métricas Dia +1: R²={r2_day1:.4f}, MAE=${mae_day1:.4f}, MSE=${mse_day1:.4f}")

print("\\n--- Dia +5 ---")
r2_day5 = r2_score(y_price_test_day5, y_price_pred_day5)
mae_day5 = mean_absolute_error(y_price_test_day5, y_price_pred_day5)
mse_day5 = mean_squared_error(y_price_test_day5, y_price_pred_day5)
print(f'R²:  {r2_day5:.4f}')
print(f'MAE: ${mae_day5:.4f}')
print(f'MSE: ${mse_day5:.4f}')
logging.info(f"Métricas Dia +5: R²={r2_day5:.4f}, MAE=${mae_day5:.4f}, MSE=${mse_day5:.4f}")

print("\\n--- Média para Dia +1 e Dia +5 combinados ---")
r2_combined = r2_score(combined_y_test, combined_y_pred)
mae_combined = mean_absolute_error(combined_y_test, combined_y_pred)
mse_combined = mean_squared_error(combined_y_test, combined_y_pred)
print(f'R² (Combinado):  {r2_combined:.4f}')
print(f'MAE (Combinado): ${mae_combined:.4f}')
print(f'MSE (Combinado): ${mse_combined:.4f}')
logging.info(f"Métricas Combinadas: R²={r2_combined:.4f}, MAE=${mae_combined:.4f}, MSE=${mse_combined:.4f}")

logging.info("Execução do script concluída.")


# ========== CONTROLE 1: BASELINE INGÊNUO (ÚLTIMO VALOR CONHECIDO) ==========
logging.info("=== INICIANDO ANÁLISE DO CONTROLE 1: BASELINE INGÊNUO ===")

# Baseline ingênuo: prever que o preço permanecerá constante no último valor conhecido
y_price_baseline = np.tile(last_known_prices.reshape(-1, 1), (1, forecast_horizon))

# Converter para as mesmas saídas que usamos para métricas
y_price_baseline_day1 = y_price_baseline[:, 0]
y_price_baseline_day5 = y_price_baseline[:, 4]

logging.info("Calculando métricas do controle 1 (baseline ingênuo).")
print("\\n========== MÉTRICAS DE DESEMPENHO - CONTROLE 1: BASELINE INGÊNUO ==========")

print("\\n--- Dia +1 ---")
r2_baseline_day1 = r2_score(y_price_test_day1, y_price_baseline_day1)
mae_baseline_day1 = mean_absolute_error(y_price_test_day1, y_price_baseline_day1)
mse_baseline_day1 = mean_squared_error(y_price_test_day1, y_price_baseline_day1)
print(f'R²:  {r2_baseline_day1:.4f}')
print(f'MAE: ${mae_baseline_day1:.4f}')
print(f'MSE: ${mse_baseline_day1:.4f}')
logging.info(f"Controle 1 - Dia +1: R²={r2_baseline_day1:.4f}, MAE=${mae_baseline_day1:.4f}, MSE=${mse_baseline_day1:.4f}")

print("\\n--- Dia +5 ---")
r2_baseline_day5 = r2_score(y_price_test_day5, y_price_baseline_day5)
mae_baseline_day5 = mean_absolute_error(y_price_test_day5, y_price_baseline_day5)
mse_baseline_day5 = mean_squared_error(y_price_test_day5, y_price_baseline_day5)
print(f'R²:  {r2_baseline_day5:.4f}')
print(f'MAE: ${mae_baseline_day5:.4f}')
print(f'MSE: ${mse_baseline_day5:.4f}')
logging.info(f"Controle 1 - Dia +5: R²={r2_baseline_day5:.4f}, MAE=${mae_baseline_day5:.4f}, MSE=${mse_baseline_day5:.4f}")

print("\\n--- Média para Dia +1 e Dia +5 combinados ---")
combined_y_baseline = np.concatenate((y_price_baseline_day1, y_price_baseline_day5))
r2_baseline_combined = r2_score(combined_y_test, combined_y_baseline)
mae_baseline_combined = mean_absolute_error(combined_y_test, combined_y_baseline)
mse_baseline_combined = mean_squared_error(combined_y_test, combined_y_baseline)
print(f'R² (Combinado):  {r2_baseline_combined:.4f}')
print(f'MAE (Combinado): ${mae_baseline_combined:.4f}')
print(f'MSE (Combinado): ${mse_baseline_combined:.4f}')
logging.info(f"Controle 1 - Combinado: R²={r2_baseline_combined:.4f}, MAE=${mae_baseline_combined:.4f}, MSE=${mse_baseline_combined:.4f}")

# Gráficos do controle 1 (opcional, para comparação visual)
plt.figure(figsize=(8, 6))
plt.scatter(y_price_test_day1, y_price_baseline_day1, alpha=0.5, color='blue')
plt.plot([min(y_price_test_day1), max(y_price_test_day1)],
         [min(y_price_test_day1), max(y_price_test_day1)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +1 Real vs Previsto (Café) - Controle 1: Baseline Ingênuo')
plt.xlabel('Real (USD)')
plt.ylabel('Previsto (USD)')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_1_baseline.png')

plt.figure(figsize=(8, 6))
plt.scatter(y_price_test_day5, y_price_baseline_day5, alpha=0.5, color='green')
plt.plot([min(y_price_test_day5), max(y_price_test_day5)],
         [min(y_price_test_day5), max(y_price_test_day5)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +5 Real vs Previsto (Café) - Controle 1: Baseline Ingênuo')
plt.xlabel('Real (USD)')
plt.ylabel('Previsto (USD)')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_5_baseline.png')


# ========== CONTROLE 2: LSTM COM ENTRADA ÚNICA (APENAS CAFÉ) ==========
logging.info("=== INICIANDO ANÁLISE DO CONTROLE 2: LSTM COM ENTRADA ÚNICA (CAFÉ) ===")

# Preparar dados para o controle 2: apenas a variação do café como feature
feature_names_control2 = ['Var_Cafe']
features_data_control2 = df[feature_names_control2].values

# Normalizar
scaler_features_control2 = MinMaxScaler()
scaled_features_control2 = scaler_features_control2.fit_transform(features_data_control2)

# Usar o mesmo target (variação do café) e mesmo scaler do target
# Criar janelas
X_control2, y_control2 = [], []
for i in range(len(scaled_features_control2) - window_size - forecast_horizon + 1):
    X_control2.append(scaled_features_control2[i:i+window_size])
    y_control2.append(scaled_target[i+window_size:i+window_size+forecast_horizon])

X_control2 = np.array(X_control2)
y_control2 = np.array(y_control2)
logging.info(f"Controle 2 - Janelas criadas. X shape: {X_control2.shape}, y shape: {y_control2.shape}")

# Dividir dados (mesmos splits que antes para comparação justa)
X_train_c2, y_train_c2 = X_control2[:train_split], y_control2[:train_split]
X_val_c2, y_val_c2 = X_control2[train_split:val_split], y_control2[train_split:val_split]
X_test_c2, y_test_c2 = X_control2[val_split:], y_control2[val_split:]

# Construir e treinar o modelo LSTM de entrada única
logging.info("Construindo e treinando o modelo LSTM de entrada única.")
model_control2 = Sequential()
model_control2.add(LSTM(128, return_sequences=True, input_shape=(window_size, X_control2.shape[2])))
model_control2.add(Dropout(0.2))
model_control2.add(LSTM(64))
model_control2.add(Dense(forecast_horizon))

model_control2.compile(optimizer='adam', loss='mse')
history_control2 = model_control2.fit(X_train_c2, y_train_c2, epochs=50, batch_size=32,
                                      validation_data=(X_val_c2, y_val_c2), verbose=1)
logging.info("Modelo LSTM de entrada única treinado.")

# Previsões
predicted_scaled_c2 = model_control2.predict(X_test_c2)
y_rescaled_c2 = scaler_target.inverse_transform(y_test_c2)
predicted_rescaled_c2 = scaler_target.inverse_transform(predicted_scaled_c2)

# Converter para preços (mesmo processo que antes)
y_price_pred_c2 = np.zeros_like(y_rescaled_c2)
y_price_true_c2 = np.zeros_like(y_rescaled_c2)  # Deve ser o mesmo que y_price_true do modelo principal

for i in range(len(X_test_c2)):
    last_price = last_known_prices[i]  # Mesmo array de últimos preços conhecidos
    
    # Preço real futuro (mesmo que antes)
    price_true = last_price
    for j in range(forecast_horizon):
        price_true = price_true * (1 + y_rescaled_c2[i, j])
        y_price_true_c2[i, j] = price_true
    
    # Preço previsto futuro
    price_pred = last_price
    for j in range(forecast_horizon):
        price_pred = price_pred * (1 + predicted_rescaled_c2[i, j])
        y_price_pred_c2[i, j] = price_pred

# Extrair Dia +1 e Dia +5
y_price_test_day1_c2 = y_price_true_c2[:, 0]
y_price_pred_day1_c2 = y_price_pred_c2[:, 0]
y_price_test_day5_c2 = y_price_true_c2[:, 4]
y_price_pred_day5_c2 = y_price_pred_c2[:, 4]

logging.info("Calculando métricas do controle 2 (LSTM entrada única).")
print("\\n========== MÉTRICAS DE DESEMPENHO - CONTROLE 2: LSTM ENTRADA ÚNICA (CAFÉ) ==========")

print("\\n--- Dia +1 ---")
r2_c2_day1 = r2_score(y_price_test_day1_c2, y_price_pred_day1_c2)
mae_c2_day1 = mean_absolute_error(y_price_test_day1_c2, y_price_pred_day1_c2)
mse_c2_day1 = mean_squared_error(y_price_test_day1_c2, y_price_pred_day1_c2)
print(f'R²:  {r2_c2_day1:.4f}')
print(f'MAE: ${mae_c2_day1:.4f}')
print(f'MSE: ${mse_c2_day1:.4f}')
logging.info(f"Controle 2 - Dia +1: R²={r2_c2_day1:.4f}, MAE=${mae_c2_day1:.4f}, MSE=${mse_c2_day1:.4f}")

print("\\n--- Dia +5 ---")
r2_c2_day5 = r2_score(y_price_test_day5_c2, y_price_pred_day5_c2)
mae_c2_day5 = mean_absolute_error(y_price_test_day5_c2, y_price_pred_day5_c2)
mse_c2_day5 = mean_squared_error(y_price_test_day5_c2, y_price_pred_day5_c2)
print(f'R²:  {r2_c2_day5:.4f}')
print(f'MAE: ${mae_c2_day5:.4f}')
print(f'MSE: ${mse_c2_day5:.4f}')
logging.info(f"Controle 2 - Dia +5: R²={r2_c2_day5:.4f}, MAE=${mae_c2_day5:.4f}, MSE=${mse_c2_day5:.4f}")

print("\\n--- Média para Dia +1 e Dia +5 combinados ---")
combined_y_test_c2 = np.concatenate((y_price_test_day1_c2, y_price_test_day5_c2))
combined_y_pred_c2 = np.concatenate((y_price_pred_day1_c2, y_price_pred_day5_c2))
r2_c2_combined = r2_score(combined_y_test_c2, combined_y_pred_c2)
mae_c2_combined = mean_absolute_error(combined_y_test_c2, combined_y_pred_c2)
mse_c2_combined = mean_squared_error(combined_y_test_c2, combined_y_pred_c2)
print(f'R² (Combinado):  {r2_c2_combined:.4f}')
print(f'MAE (Combinado): ${mae_c2_combined:.4f}')
print(f'MSE (Combinado): ${mse_c2_combined:.4f}')
logging.info(f"Controle 2 - Combinado: R²={r2_c2_combined:.4f}, MAE=${mae_c2_combined:.4f}, MSE=${mse_c2_combined:.4f}")

# Gráficos do controle 2
plt.figure(figsize=(8, 6))
plt.scatter(y_price_test_day1_c2, y_price_pred_day1_c2, alpha=0.5, color='blue')
plt.plot([min(y_price_test_day1_c2), max(y_price_test_day1_c2)],
         [min(y_price_test_day1_c2), max(y_price_test_day1_c2)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +1 Real vs Previsto (Café) - Controle 2: LSTM Entrada Única')
plt.xlabel('Real (USD)')
plt.ylabel('Previsto (USD)')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_1_entr_unica.png')

plt.figure(figsize=(8, 6))
plt.scatter(y_price_test_day5_c2, y_price_pred_day5_c2, alpha=0.5, color='green')
plt.plot([min(y_price_test_day5_c2), max(y_price_test_day5_c2)],
         [min(y_price_test_day5_c2), max(y_price_test_day5_c2)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +5 Real vs Previsto (Café) - Controle 2: LSTM Entrada Única')
plt.xlabel('Real (USD)')
plt.ylabel('Previsto (USD)')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_5_entr_unica.png')

logging.info("Análise dos controles concluída.")