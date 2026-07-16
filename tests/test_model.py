from pathlib import Path

import joblib
import numpy as np


RAIZ_PROYECTO = Path(
    __file__
).resolve().parent.parent

RUTA_MODELO = (
    RAIZ_PROYECTO
    / "models"
    / "modelo_sentimiento_v2.joblib"
)


def test_modelo_existe():
    assert RUTA_MODELO.exists()


def test_modelo_predice():
    modelo = joblib.load(
        RUTA_MODELO
    )

    resultado = modelo.predict(
        [
            "Apple reports strong "
            "quarterly results"
        ]
    )

    assert resultado[0] in {
        "negative",
        "neutral",
        "positive",
    }


def test_probabilidades_suman_uno():
    modelo = joblib.load(
        RUTA_MODELO
    )

    probabilidades = (
        modelo.predict_proba(
            [
                "Apple reports strong "
                "quarterly results"
            ]
        )[0]
    )

    assert np.isclose(
        probabilidades.sum(),
        1.0,
    )