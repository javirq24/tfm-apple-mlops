import json
from pathlib import Path

import joblib
import nltk
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import TextCleaner


SEMILLA = 42
CLASES = ["negative", "neutral", "positive"]

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ_PROYECTO / "data" / "resultado_actual.xlsx"
CARPETA_MODELOS = RAIZ_PROYECTO / "models"

RUTA_MODELO = (
    CARPETA_MODELOS / "modelo_sentimiento_v2.joblib"
)

RUTA_METRICAS = (
    CARPETA_MODELOS / "metricas_modelo_v2.json"
)


def cargar_y_validar_datos() -> pd.DataFrame:
    """
    Carga el Excel y aplica controles básicos de calidad.
    """

    if not RUTA_DATOS.exists():
        raise FileNotFoundError(
            "No se ha encontrado el fichero de entrenamiento en:\n"
            f"{RUTA_DATOS}"
        )

    df = pd.read_excel(
        RUTA_DATOS,
        sheet_name="Hoja1",
    )

    df.columns = [
        columna.strip()
        for columna in df.columns
    ]

    columnas_necesarias = {
        "Tweet",
        "Sentimiento",
    }

    columnas_faltantes = (
        columnas_necesarias
        - set(df.columns)
    )

    if columnas_faltantes:
        raise ValueError(
            "Faltan las siguientes columnas obligatorias: "
            f"{sorted(columnas_faltantes)}"
        )

    df["Tweet"] = (
        df["Tweet"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["Sentimiento"].isin(CLASES)
    ]

    df = df[
        df["Tweet"].str.len() > 15
    ]

    df = df.drop_duplicates(
        subset="Tweet",
        keep="first",
    )

    if df.empty:
        raise ValueError(
            "No quedan registros válidos después de la limpieza."
        )

    clases_encontradas = set(
        df["Sentimiento"].unique()
    )

    clases_faltantes = (
        set(CLASES)
        - clases_encontradas
    )

    if clases_faltantes:
        raise ValueError(
            "No existen registros para estas clases: "
            f"{sorted(clases_faltantes)}"
        )

    return df


def crear_modelo() -> Pipeline:
    """
    Crea el pipeline completo:
    limpieza + TF-IDF + regresión logística.
    """

    modelo = Pipeline(
        steps=[
            (
                "limpieza",
                TextCleaner(),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "clasificador",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=3.0,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )

    return modelo


def main() -> None:
    nltk.download(
        "stopwords",
        quiet=True,
    )

    CARPETA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Cargando y validando los datos...")

    df = cargar_y_validar_datos()

    print(
        f"Registros válidos: {len(df)}"
    )

    print("\nDistribución de clases:")
    print(
        df["Sentimiento"].value_counts()
    )

    X = df["Tweet"].values
    y = df["Sentimiento"].values

    (
        X_entrenamiento,
        X_prueba,
        y_entrenamiento,
        y_prueba,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEMILLA,
        stratify=y,
    )

    print("\nCreando y entrenando el modelo...")

    modelo = crear_modelo()

    modelo.fit(
        X_entrenamiento,
        y_entrenamiento,
    )

    predicciones = modelo.predict(
        X_prueba
    )

    accuracy = accuracy_score(
        y_prueba,
        predicciones,
    )

    f1_macro = f1_score(
        y_prueba,
        predicciones,
        average="macro",
    )

    matriz = confusion_matrix(
        y_prueba,
        predicciones,
        labels=CLASES,
    )

    reporte = classification_report(
        y_prueba,
        predicciones,
        labels=CLASES,
        zero_division=0,
        output_dict=True,
    )

    metricas = {
        "version_modelo": "2",
        "semilla": SEMILLA,
        "algoritmo": "LogisticRegression",
        "vectorizador": "TfidfVectorizer",
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "registros_totales": int(len(df)),
        "registros_entrenamiento": int(
            len(X_entrenamiento)
        ),
        "registros_prueba": int(
            len(X_prueba)
        ),
        "matriz_confusion": matriz.tolist(),
        "reporte_clasificacion": reporte,
    }

    joblib.dump(
        modelo,
        RUTA_MODELO,
    )

    with open(
        RUTA_METRICAS,
        "w",
        encoding="utf-8",
    ) as archivo_metricas:
        json.dump(
            metricas,
            archivo_metricas,
            indent=4,
        )

    print("\nEntrenamiento finalizado.")
    print(
        f"Accuracy: {accuracy:.4f}"
    )
    print(
        f"F1-macro: {f1_macro:.4f}"
    )

    print("\nMatriz de confusión:")
    print(matriz)

    print(
        "\nModelo guardado en:\n"
        f"{RUTA_MODELO}"
    )

    print(
        "\nMétricas guardadas en:\n"
        f"{RUTA_METRICAS}"
    )


if __name__ == "__main__":
    main()