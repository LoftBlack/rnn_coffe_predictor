import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from gnews import GNews
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class Config:
    periodo_inicial: str = "2025-01-01"
    periodo_final: str = "2025-12-31"
    max_resultados_por_termo: int = 100
    language: str = "pt"
    country: str = "BR"
    modelo_sentimento: str = "lucas-leme/FinBERT-PT-BR"
    termos_de_busca: tuple[str, ...] = (
        "lucro",
        "ganho",
        "crescimento",
        "aumento",
        "recorde",
        "prejuizo",
        "queda",
        "recuo",
        "processo",
        "multa",
    )


def parse_date(date_str: str) -> tuple[int, int, int]:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return (date_obj.year, date_obj.month, date_obj.day)


def criar_cliente_gnews(cfg: Config) -> GNews:
    gnews = GNews(
        language=cfg.language,
        country=cfg.country,
        start_date=parse_date(cfg.periodo_inicial),
        end_date=parse_date(cfg.periodo_final),
        max_results=cfg.max_resultados_por_termo,
    )
    return gnews


def coletar_noticias(google_news: GNews, termos: Iterable[str]) -> pd.DataFrame:
    noticias_brutas = []

    for termo in termos:
        print(f"Buscando noticias para o termo: '{termo}'...")
        noticias_termo = google_news.get_news(termo)
        if noticias_termo:
            noticias_brutas.extend(noticias_termo)

    if not noticias_brutas:
        return pd.DataFrame()

    df_noticias = pd.DataFrame(noticias_brutas)
    df_noticias = df_noticias.drop_duplicates(subset=["title"]).copy()
    return df_noticias


def carregar_modelo(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return tokenizer, model


def sentimento_score(texto: str, tokenizer, model) -> float:
    if not isinstance(texto, str) or not texto.strip():
        return 0.0

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
    return float(probs[2].item() - probs[0].item())


def processar_sentimento(df_noticias: pd.DataFrame, tokenizer, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_noticias.empty:
        return df_noticias, pd.DataFrame(columns=["date", "media", "desvio_padrao"])

    df = df_noticias.copy()
    df["title"] = df.get("title", "").fillna("")
    df["description"] = df.get("description", "").fillna("")
    df["texto"] = (df["title"] + " " + df["description"]).str.strip()

    df["date"] = pd.to_datetime(df.get("published date"), errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    tqdm.pandas(desc="Analisando noticias")
    df["sentimento"] = df["texto"].progress_apply(lambda txt: sentimento_score(txt, tokenizer, model))

    media_por_dia = (
        df.groupby(df["date"].dt.date)["sentimento"]
        .agg(media="mean", desvio_padrao="std")
        .fillna(0.0)
        .reset_index()
        .sort_values("date")
    )
    media_por_dia["date"] = pd.to_datetime(media_por_dia["date"])
    return df, media_por_dia


def plotar_sentimento_diario(media_por_dia: pd.DataFrame) -> None:
    if media_por_dia.empty:
        print("Sem dados para plotar.")
        return

    plt.figure(figsize=(16, 5))
    plt.plot(media_por_dia["date"], media_por_dia["media"], marker="o", label="Sentimento medio")
    plt.fill_between(
        media_por_dia["date"],
        media_por_dia["media"] - media_por_dia["desvio_padrao"],
        media_por_dia["media"] + media_por_dia["desvio_padrao"],
        alpha=0.25,
        label="Desvio padrao",
    )
    plt.axhline(0, linewidth=1)
    plt.title("Sentimento Medio Diario das Noticias", fontsize=14)
    plt.xlabel("Data")
    plt.ylabel("Sentimento (-1 a +1)")
    plt.grid(True, linestyle="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    cfg = Config(
        periodo_inicial=os.getenv("PERIODO_INICIAL", "2025-01-01"),
        periodo_final=os.getenv("PERIODO_FINAL", "2025-12-31"),
    )

    google_news = criar_cliente_gnews(cfg)
    df_noticias = coletar_noticias(google_news, cfg.termos_de_busca)

    print(f"Total de noticias unicas: {len(df_noticias)}")
    if df_noticias.empty:
        return

    tokenizer, model = carregar_modelo(cfg.modelo_sentimento)
    df_com_sentimento, media_por_dia = processar_sentimento(df_noticias, tokenizer, model)

    print(media_por_dia)
    print(f"Existem {df_com_sentimento['date'].dt.date.nunique()} datas diferentes no banco de noticias.")
    print(df_com_sentimento["date"].dt.date.value_counts().sort_index())

    plotar_sentimento_diario(media_por_dia)


if __name__ == "__main__":
    main()
