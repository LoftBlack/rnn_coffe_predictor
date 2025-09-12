import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gnews import GNews
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ========= COLETA DE NOTÍCIAS =========
google_news = GNews(language='en', country='US', period='12m')
results = google_news.get_news('coffee price')

df = pd.DataFrame(results)
df = df[["title", "published date", "description"]]
df.rename(columns={"published date": "date"}, inplace=True)
df["date"] = pd.to_datetime(df["date"])

# ========= SENTIMENTO =========
sentiment_model = pipeline("sentiment-analysis")
df["sentiment"] = df["title"].apply(lambda x: sentiment_model(x[:512])[0]["label"])
df["score"] = df["title"].apply(lambda x: sentiment_model(x[:512])[0]["score"])

# ========= EMBEDDINGS =========
embedder = SentenceTransformer('all-MiniLM-L6-v2')
df["embedding"] = df["title"].apply(lambda x: embedder.encode(x))

# ========= PCA PARA VISUALIZAÇÃO =========
embeddings = list(df["embedding"])
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)
df["x"] = emb_2d[:,0]
df["y"] = emb_2d[:,1]

plt.scatter(df["x"], df["y"], c=df["score"], cmap="coolwarm")
plt.title("Projeção Vetorial das Notícias (PCA)")
plt.colorbar(label="Sentimento")
plt.show()

# ========= AGREGAÇÃO MENSAL =========
df["month"] = df["date"].dt.to_period("M")
monthly_sentiment = df.groupby("month")["score"].mean()

# Simulação: adicionar preços do café
coffee_prices = pd.Series(
    [120, 123, 119, 125, 128, 130, 127, 129, 135, 138, 136, 140],
    index=monthly_sentiment.index
)

# ========= MATRIZ DE CORRELAÇÃO =========
data_corr = pd.DataFrame({
    "sentiment": monthly_sentiment,
    "coffee_price": coffee_prices
})
corr_matrix = data_corr.corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlação Sentimento vs Preço do Café")
plt.show()
