import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import nltk
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

NLTK_DATA_PATH = PROJECT_ROOT / "nltk_data"

os.environ["NLTK_DATA"] = str(NLTK_DATA_PATH)

if str(NLTK_DATA_PATH) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_PATH))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Necesario para deserializar TextCleaner desde el joblib.
from src import preprocess  # noqa: F401


model = None


def init() -> None:
    """
    Azure ejecuta esta función una sola vez al iniciar el contenedor.
    """
    global model

    nltk_data_path = PROJECT_ROOT / "nltk_data"

    os.environ["NLTK_DATA"] = str(nltk_data_path)

    if str(nltk_data_path) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_data_path))

    model_directory = os.getenv("AZUREML_MODEL_DIR")

    if not model_directory:
        raise RuntimeError(
            "No se ha definido la variable AZUREML_MODEL_DIR."
        )

    model_root = Path(model_directory)

    model_files = list(
        model_root.rglob("modelo_sentimiento_v2.joblib")
    )

    if not model_files:
        raise FileNotFoundError(
            "No se encontró modelo_sentimiento_v2.joblib "
            f"dentro de {model_root}"
        )

    model_path = model_files[0]

    model = joblib.load(model_path)

    print(f"Modelo cargado desde: {model_path}")
    print(f"Pasos del pipeline: {list(model.named_steps.keys())}")
    print(f"Clases: {list(model.classes_)}")


def run(raw_data: str) -> dict[str, Any]:
    """
    Azure ejecuta esta función para cada petición recibida.
    """
    try:
        if model is None:
            raise RuntimeError(
                "El modelo no ha sido inicializado correctamente."
            )

        payload = json.loads(raw_data)
        texts = payload.get("texts")

        if not isinstance(texts, list):
            return {
                "error": "El campo 'texts' debe contener una lista."
            }

        if not texts:
            return {
                "error": "La lista 'texts' no puede estar vacía."
            }

        if not all(
            isinstance(text, str) and text.strip()
            for text in texts
        ):
            return {
                "error": (
                    "Todos los elementos de 'texts' deben ser "
                    "cadenas de texto no vacías."
                )
            }

        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)

        results = []

        for text, prediction, probability_values in zip(
            texts,
            predictions,
            probabilities,
        ):
            probability_dictionary = {
                class_name: round(float(probability), 6)
                for class_name, probability in zip(
                    model.classes_,
                    probability_values,
                )
            }

            results.append(
                {
                    "text": text,
                    "sentiment": str(prediction),
                    "probabilities": probability_dictionary,
                    "confidence": round(
                        float(np.max(probability_values)),
                        6,
                    ),
                }
            )

        return {"predictions": results}

    except json.JSONDecodeError:
        return {
            "error": "InvalidJSON",
            "message": "El cuerpo recibido no contiene un JSON válido.",
        }

    except Exception as error:
        return {
            "error": type(error).__name__,
            "message": str(error),
        }