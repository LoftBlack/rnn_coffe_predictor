import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gnews import GNews
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import yfinance as yf

# ========= COLETA DE NOTÍCIAS =========
print("Coletando notícias sobre café...")
google_news = GNews(language='en', country='US', period='12m')
results = google_news.get_news('coffee price')

# O nome da coluna 'published date' mudou para 'published'
df = pd.DataFrame(results)
df = df[["title", "published", "description"]]
df.rename(columns={"published": "date"}, inplace=True)
df["date"] = pd.to_datetime(df["date"])

# ========= ANÁLISE DE SENTIMENTO =========
print("Analisando sentimento das notícias...")
sentiment_model = pipeline("sentiment-analysis")
df["sentiment"] = df["title"].apply(lambda x: sentiment_model(x[:512])[0]["label"])
df["score"] = df["title"].apply(lambda x: sentiment_model(x[:512])[0]["score"])

# ========= EMBEDDINGS E PCA =========
print("Calculando embeddings e PCA para visualização...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
df["embedding"] = df["title"].apply(lambda x: embedder.encode(x))

embeddings = list(df["embedding"])
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)
df["x"] = emb_2d[:,0]
df["y"] = emb_2d[:,1]

# ========= PLOT PCA =========
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df["x"], df["y"], c=df["score"], cmap="coolwarm")
plt.title("Projeção Vetorial das Notícias (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(scatter, label="Sentimento (Negativo a Positivo)")
plt.grid(True)
plt.show()

# ========= COLETA DE PREÇOS REAIS DO CAFÉ =========
# Usando yfinance para obter dados do contrato futuro de café (KC=F)
print("Coletando preços históricos do café (KC=F)...")
coffee_ticker = yf.Ticker("KC=F")
# Obtém os preços mensais dos últimos 12 meses
coffee_data = coffee_ticker.history(period="12mo", interval="1mo")
# Apenas o preço de fechamento
coffee_prices_yf = coffee_data["Close"]
# Remove o fuso horário e converte para o formato de período mensal
coffee_prices_yf.index = coffee_prices_yf.index.tz_localize(None).to_period("M")

# ========= AGREGAÇÃO E CORRELAÇÃO =========
df["month"] = df["date"].dt.to_period("M")
monthly_sentiment = df.groupby("month")["score"].mean()

# Alinhe os dados de sentimento e preço pelo índice de mês
data_corr = pd.DataFrame({
    "sentiment": monthly_sentiment,
    "coffee_price": coffee_prices_yf
}).dropna() # Remove meses sem dados

print("\nDados para correlação:")
print(data_corr)

corr_matrix = data_corr.corr()

# ========= PLOT HEATMAP DA CORRELAÇÃO =========
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação Sentimento vs Preço do Café")
plt.show()