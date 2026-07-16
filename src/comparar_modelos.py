from pathlib import Path

import joblib


RAIZ_PROYECTO = Path(
    __file__
).resolve().parent.parent

RUTA_MODELO_ANTIGUO = (
    RAIZ_PROYECTO
    / "models"
    / "modelo_sentimiento.joblib"
)

RUTA_MODELO_NUEVO = (
    RAIZ_PROYECTO
    / "models"
    / "modelo_sentimiento_v2.joblib"
)


def main() -> None:
    modelo_antiguo = joblib.load(
        RUTA_MODELO_ANTIGUO
    )

    modelo_nuevo = joblib.load(
        RUTA_MODELO_NUEVO
    )

    textos = [
        "Apple reports record revenue and strong sales",
        "Apple faces serious regulatory problems",
        "Apple releases a routine software update",
        "Investors are disappointed with Apple's guidance",
        "Apple launches innovative artificial intelligence features",
    ]

    print(
        "COMPARACIÓN ENTRE MODELOS"
    )

    for texto in textos:
        pred_antigua = (
            modelo_antiguo.predict(
                [texto]
            )[0]
        )

        pred_nueva = (
            modelo_nuevo.predict(
                [texto]
            )[0]
        )

        print("\nTexto:")
        print(texto)

        print(
            "Modelo antiguo:",
            pred_antigua,
        )

        print(
            "Modelo nuevo:",
            pred_nueva,
        )


if __name__ == "__main__":
    main()