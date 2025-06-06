import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ========== 1. DOWNLOAD DOS DADOS ==========
# Tickers:
# Café: 'KC=F'
# Petróleo: 'CL=F'
# Milho: 'ZC=F'

start_date = "2010-01-01"
end_date = "2024-12-31"

df_cafe = yf.download('KC=F', start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Cafe'})
df_petroleo = yf.download('CL=F', start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Petroleo'})
df_milho = yf.download('ZC=F', start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Milho'})

# ========== 2. UNIFICAÇÃO E FEATURES ADICIONAIS ==========
# Junta os dados em um único DataFrame com o mesmo índice
df = df_cafe.join([df_petroleo, df_milho], how='inner')

# Calcula as taxas de variação diária
df['Var_Cafe'] = df['Cafe'].pct_change()
df['Var_Petroleo'] = df['Petroleo'].pct_change()

# Simula dados climáticos (exemplo)
# Em um caso real, esses dados viriam de uma API meteorológica
np.random.seed(42)
df['Clima'] = np.random.normal(loc=0.5, scale=0.1, size=len(df))

# Remove linhas com valores NaN
df.dropna(inplace=True)

# ========== 3. NORMALIZAÇÃO ==========
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Armazena colunas para referência futura
feature_names = df.columns.tolist()

# ========== 4. CRIAÇÃO DAS JANELAS (30 DIAS → 5 DIAS) ==========
window_size = 30
forecast_horizon = 5

X, y = [], []
for i in range(len(scaled_data) - window_size - forecast_horizon):
    X.append(scaled_data[i:i+window_size])  # shape: (30, n_features)
    y.append(scaled_data[i+window_size:i+window_size+forecast_horizon, 0])  # Apenas preço do café (primeira coluna)

X = np.array(X)  # (amostras, 30, n_features)
y = np.array(y)  # (amostras, 5)

# ========== 5. DIVISÃO TREINO/TESTE ==========
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ========== 6. MODELO LSTM ==========
# Arquitetura com uma camada LSTM maior e dropout para evitar overfitting
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(window_size, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(forecast_horizon))  # Saída para 5 dias

model.compile(optimizer='adam', loss='mse')

# ========== 7. TREINAMENTO ==========
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1)

# ========== 8. PREVISÃO ==========
predicted = model.predict(X_test)

# ========== 9. DESNORMALIZAÇÃO ==========
# Inversão apenas da primeira feature (preço do café)
# Cafe é a primeira coluna do DataFrame original, logo:
cafe_index = 0

# Recria o scaler só com a coluna 'Cafe'
scaler_cafe = MinMaxScaler()
scaler_cafe.min_ = scaler.min_[cafe_index:cafe_index+1]
scaler_cafe.scale_ = scaler.scale_[cafe_index:cafe_index+1]

# Desnormaliza
y_test_rescaled = scaler_cafe.inverse_transform(y_test)
predicted_rescaled = scaler_cafe.inverse_transform(predicted)

# ========== 10. ORGANIZAÇÃO DAS SAÍDAS ==========
y_test_day1 = y_test_rescaled[:, 0]
y_pred_day1 = predicted_rescaled[:, 0]
y_test_day5 = y_test_rescaled[:, 4]
y_pred_day5 = predicted_rescaled[:, 4]

# ========== 11. CRIAÇÃO DO DIRETÓRIO DE GRÁFICOS ==========
os.makedirs("graficos", exist_ok=True)

# Função auxiliar para salvar gráficos
def salvar_grafico(nome):
    plt.tight_layout()
    plt.savefig(f'graficos/{nome}')
    plt.show()

# ========== 12. GRÁFICO DE DISPERSÃO: DIA +1 ==========
plt.figure(figsize=(8, 6))
plt.scatter(y_test_day1, y_pred_day1, alpha=0.5, color='blue')
plt.plot([min(y_test_day1), max(y_test_day1)],
         [min(y_test_day1), max(y_test_day1)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +1 Real vs Previsto')
plt.xlabel('Real')
plt.ylabel('Previsto')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_1.png')

# ========== 13. GRÁFICO DE DISPERSÃO: DIA +5 ==========
plt.figure(figsize=(8, 6))
plt.scatter(y_test_day5, y_pred_day5, alpha=0.5, color='green')
plt.plot([min(y_test_day5), max(y_test_day5)],
         [min(y_test_day5), max(y_test_day5)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +5 Real vs Previsto')
plt.xlabel('Real')
plt.ylabel('Previsto')
plt.legend()
plt.grid(True)
salvar_grafico('dispersao_dia_5.png')

# ========== 14. COMPARAÇÃO TEMPORAL: DIA +1 ==========
plt.figure(figsize=(12, 5))
plt.plot(y_test_day1, label='Real (Dia +1)', color='blue')
plt.plot(y_pred_day1, label='Previsto (Dia +1)', color='red', linestyle='--')
plt.title('Comparação Temporal - Dia +1')
plt.xlabel('Amostra')
plt.ylabel('Preço')
plt.legend()
plt.grid(True)
salvar_grafico('comparacao_temporal_dia_1.png')

# ========== 15. COMPARAÇÃO TEMPORAL: DIA +5 ==========
plt.figure(figsize=(12, 5))
plt.plot(y_test_day5, label='Real (Dia +5)', color='blue')
plt.plot(y_pred_day5, label='Previsto (Dia +5)', color='orange', linestyle='--')
plt.title('Comparação Temporal - Dia +5')
plt.xlabel('Amostra')
plt.ylabel('Preço')
plt.legend()
plt.grid(True)
salvar_grafico('comparacao_temporal_dia_5.png')

# ========== 16. HISTÓRICO DE PERDA ==========
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Histórico de Perda (Loss)')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
salvar_grafico('historico_perda.png')

# ========== 17. MÉTRICAS ==========
print("\n========== MÉTRICAS DE DESEMPENHO ==========")

print("\n--- Dia +1 ---")
r2_day1 = r2_score(y_test_day1, y_pred_day1)
mae_day1 = mean_absolute_error(y_test_day1, y_pred_day1)
mse_day1 = mean_squared_error(y_test_day1, y_pred_day1)
print(f'R²:  {r2_day1:.4f}')
print(f'MAE: ${mae_day1:.4f}')
print(f'MSE: ${mse_day1:.4f}')

print("\n--- Dia +5 ---")
r2_day5 = r2_score(y_test_day5, y_pred_day5)
mae_day5 = mean_absolute_error(y_test_day5, y_pred_day5)
mse_day5 = mean_squared_error(y_test_day5, y_pred_day5)
print(f'R²:  {r2_day5:.4f}')
print(f'MAE: ${mae_day5:.4f}')
print(f'MSE: ${mse_day5:.4f}')
