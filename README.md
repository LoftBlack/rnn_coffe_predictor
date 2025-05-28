# Coffee Price Forecasting with RNN

Este projeto utiliza uma Rede Neural Recorrente (RNN) com LSTM para prever o preço futuro do café (Café Arábica - `KC=F`) com até 5 dias de antecedência. O modelo é treinado com séries temporais financeiras e pode futuramente incluir variáveis como preços do petróleo, milho e dados climáticos.

## 📊 Tecnologias Usadas

- Python
- TensorFlow / Keras
- Pandas / NumPy / Matplotlib
- yFinance (para obtenção dos dados históricos)

## 🧠 Objetivo

Prever os preços do café para os próximos 5 dias, utilizando uma janela deslizante de 30 dias anteriores. O modelo é avaliado com gráficos de comparação e métricas como R², MAE e MSE.

## 🚀 Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/LoftBlack/rnn_coffe_predictor.git
   cd rnn_coffe_predictor

  
