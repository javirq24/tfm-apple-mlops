import re
from typing import Iterable

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin


STOP_WORDS = set(stopwords.words("english"))

PALABRAS_RUIDO = {
    "apple",
    "aapl",
    "iphone",
    "ipad",
    "ipod",
    "itunes",
    "ios",
    "mac",
    "macbook",
    "rt",
    "amp",
    "https",
    "http",
    "co",
    "new",
    "news",
    "says",
    "report",
}

STEMMER = SnowballStemmer("english")


def limpiar_texto(texto: str) -> str:
    """
    Limpia un texto antes de aplicar TF-IDF.

    Operaciones:
    - convierte a minúsculas;
    - elimina URLs;
    - elimina menciones;
    - conserva el contenido de los hashtags;
    - elimina números y signos;
    - elimina palabras vacías;
    - elimina vocabulario de ruido;
    - aplica stemming.
    """

    texto = str(texto).lower()

    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"@\w+", " ", texto)
    texto = re.sub(r"#(\w+)", r"\1", texto)
    texto = re.sub(r"[^a-záéíóúñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    palabras_filtradas = []

    for palabra in texto.split():
        if palabra in STOP_WORDS:
            continue

        if palabra in PALABRAS_RUIDO:
            continue

        if len(palabra) <= 2:
            continue

        palabra_reducida = STEMMER.stem(palabra)
        palabras_filtradas.append(palabra_reducida)

    return " ".join(palabras_filtradas)


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Transformador compatible con los Pipeline de Scikit-learn.
    """

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]):
        return [limpiar_texto(texto) for texto in X]