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
   ```
   git clone https://github.com/LoftBlack/rnn_coffe_predictor.git
   cd rnn_coffe_predictor
2. Crie e ative um ambiente virtual:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate    # Windows
3.Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
4.Execute o script principal:
```
   python main.py
```
📈 Resultados
O script gera gráficos que mostram a comparação entre os preços reais e previstos, tanto em série temporal quanto em dispersão para os dias 1 e 5 da previsão.
