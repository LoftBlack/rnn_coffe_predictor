import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Baixar dados da comodity (ex: café futuro)
ticker = 'KC=F'  # Café arábica futuro
data = yf.download(ticker, start="2010-01-01", end="2024-12-31")

# 2. Selecionar a coluna correta
if 'Adj Close' in data.columns:
    prices = data[['Adj Close']].dropna()
else:
    prices = data[['Close']].dropna()

# 3. Normalizar
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(prices)

# 4. Criar janelas (30 dias → prever próximos 5)
window_size = 30
forecast_horizon = 5

X, y = [], []
for i in range(len(normalized_data) - window_size - forecast_horizon):
    X.append(normalized_data[i:i + window_size])
    y.append(normalized_data[i + window_size:i + window_size + forecast_horizon].flatten())

X = np.array(X)
y = np.array(y)

# 5. Dividir em treino/teste (80/20)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 6. Construir o modelo
model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(window_size, 1)))
model.add(Dense(forecast_horizon))  # 5 saídas
model.compile(optimizer='adam', loss='mse')

# 7. Treinar
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 8. Fazer previsões
predicted = model.predict(X_test)

# 9. Desnormalizar
def desnormalizar_lote(lote):
    preenchido = np.concatenate((lote, np.zeros((len(lote), 1))), axis=1)
    return scaler.inverse_transform(preenchido)[:, :forecast_horizon]

y_test_rescaled = desnormalizar_lote(y_test)
predicted_rescaled = desnormalizar_lote(predicted)

# 10. Extrair dia +1 e +5
y_test_day1 = y_test_rescaled[:, 0]
y_pred_day1 = predicted_rescaled[:, 0]
y_test_day5 = y_test_rescaled[:, 4]
y_pred_day5 = predicted_rescaled[:, 4]

# 11. Gráfico de Dispersão - Dia +1
plt.figure(figsize=(8, 6))
plt.scatter(y_test_day1, y_pred_day1, alpha=0.5, color='blue')
plt.plot([min(y_test_day1), max(y_test_day1)],
         [min(y_test_day1), max(y_test_day1)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +1 Real vs Previsto')
plt.xlabel('Real')
plt.ylabel('Previsto')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Gráfico de Dispersão - Dia +5
plt.figure(figsize=(8, 6))
plt.scatter(y_test_day5, y_pred_day5, alpha=0.5, color='green')
plt.plot([min(y_test_day5), max(y_test_day5)],
         [min(y_test_day5), max(y_test_day5)], 'r--', label='Ideal')
plt.title('Dispersão: Dia +5 Real vs Previsto')
plt.xlabel('Real')
plt.ylabel('Previsto')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 13. Gráfico: Real vs Previsto ao longo do tempo - Dia +1
plt.figure(figsize=(12, 5))
plt.plot(y_test_day1, label='Real (Dia +1)', color='blue')
plt.plot(y_pred_day1, label='Previsto (Dia +1)', color='red', linestyle='--')
plt.title('Comparação Temporal - Dia +1')
plt.xlabel('Amostra')
plt.ylabel('Preço')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 14. Gráfico: Real vs Previsto ao longo do tempo - Dia +5
plt.figure(figsize=(12, 5))
plt.plot(y_test_day5, label='Real (Dia +5)', color='blue')
plt.plot(y_pred_day5, label='Previsto (Dia +5)', color='orange', linestyle='--')
plt.title('Comparação Temporal - Dia +5')
plt.xlabel('Amostra')
plt.ylabel('Preço')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 15. Histórico da perda
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Histórico de Perda (Loss)')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 16. Métricas de Desempenho - Dia +1
print("========== Dia +1 ==========")
r2_day1 = r2_score(y_test_day1, y_pred_day1)
mae_day1 = mean_absolute_error(y_test_day1, y_pred_day1)
mse_day1 = mean_squared_error(y_test_day1, y_pred_day1)
print(f'R²: {r2_day1:.4f}')
print(f'MAE: ${mae_day1:.4f}')
print(f'MSE: ${mse_day1:.4f}')

# 17. Métricas de Desempenho - Dia +5
print("\n========== Dia +5 ==========")
r2_day5 = r2_score(y_test_day5, y_pred_day5)
mae_day5 = mean_absolute_error(y_test_day5, y_pred_day5)
mse_day5 = mean_squared_error(y_test_day5, y_pred_day5)
print(f'R²: {r2_day5:.4f}')
print(f'MAE: ${mae_day5:.4f}')
print(f'MSE: ${mse_day5:.4f}')
