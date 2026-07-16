import json
import os
from pathlib import Path

from src import score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY = PROJECT_ROOT / "models"


def inicializar_modelo() -> None:
    os.environ["AZUREML_MODEL_DIR"] = str(MODEL_DIRECTORY)
    score.init()


def test_score_devuelve_predicciones():
    inicializar_modelo()

    request = json.dumps(
        {
            "texts": [
                "Apple reports strong quarterly results"
            ]
        }
    )

    response = score.run(request)

    assert "predictions" in response
    assert len(response["predictions"]) == 1

    prediction = response["predictions"][0]

    assert prediction["sentiment"] in {
        "negative",
        "neutral",
        "positive",
    }

    assert "probabilities" in prediction
    assert "confidence" in prediction


def test_score_rechaza_lista_vacia():
    inicializar_modelo()

    response = score.run(
        json.dumps({"texts": []})
    )

    assert "error" in response


def test_score_rechaza_json_invalido():
    inicializar_modelo()

    response = score.run(
        "esto no es un json"
    )

    assert response["error"] == "InvalidJSON"