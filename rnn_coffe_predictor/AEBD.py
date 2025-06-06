# 1. Imports
import os
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from dotenv import load_dotenv
import os

load_dotenv()  # carrega o .env automaticamente da raiz
NOAA_TOKEN = os.getenv("NOAA_TOKEN")

# 2. Baixar preço do café
df_cafe = yf.download('KC=F', start='2010-01-01', end='2024-12-31')[['Close']]
df_cafe.rename(columns={'Close': 'PrecoCafe'}, inplace=True)

# 3. Configurar NOAA API
headers = {"token": NOAA_TOKEN}
station_id = "GHCND:USW00094728"  # Estação em Nova York (JFK), você pode mudar para outra se quiser
start_date = "2010-01-01"
end_date = "2024-12-31"

# 4. Função para baixar dados do NOAA
def fetch_noaa(datatypeid):
    url = (
        f"https://www.ncei.noaa.gov/cdo-web/api/v2/data"
        f"?datasetid=GHCND&stationid={station_id}"
        f"&datatypeid={datatypeid}&startdate={start_date}"
        f"&enddate={end_date}&limit=1000&units=metric"
    )
    resp = requests.get(url, headers=headers)
    print(resp.status_code,"macaco")
    print(resp.json())  # <-- Adicione esta linha

    data = resp.json().get('results', [])
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[['date', 'datatype', 'value']]
    df = df.pivot(index='date', columns='datatype', values='value')
    return df

# 5. Baixar dados de temperatura e precipitação
df_temp = fetch_noaa('TAVG')
df_prcp = fetch_noaa('PRCP')

# 6. Unir dados
df_temp.index = pd.to_datetime(df_temp.index)
df_prcp.index = pd.to_datetime(df_prcp.index)
df_clima = df_temp.join(df_prcp, how='outer')
df_all = df_cafe.join(df_clima, how='inner').dropna()

# 7. Matriz de correlação
corr = df_all.corr()
print("Correlação com Preço do Café:")
print(corr['PrecoCafe'].sort_values(ascending=False))

# 8. Visualizar como heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação (Café + Clima)')
plt.tight_layout()
plt.savefig('graficos/correlacao_clima_cafe.png')
plt.show()
