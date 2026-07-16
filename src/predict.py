from pathlib import Path

import joblib


RAIZ_PROYECTO = Path(
    __file__
).resolve().parent.parent

RUTA_MODELO = (
    RAIZ_PROYECTO
    / "models"
    / "modelo_sentimiento_v2.joblib"
)


def main() -> None:
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            "No se encuentra el modelo nuevo. "
            "Ejecuta primero: python src/train.py"
        )

    modelo = joblib.load(
        RUTA_MODELO
    )

    print(
        "MODELO DE ANÁLISIS DE SENTIMIENTOS"
    )

    print(
        "Escribe un texto en inglés."
    )

    texto = input(
        "\nTexto: "
    ).strip()

    if not texto:
        print(
            "No se ha introducido ningún texto."
        )
        return

    prediccion = modelo.predict(
        [texto]
    )[0]

    probabilidades = (
        modelo.predict_proba(
            [texto]
        )[0]
    )

    print("\nResultado")
    print(
        "Sentimiento:",
        prediccion,
    )

    print("\nProbabilidades:")

    for clase, probabilidad in zip(
        modelo.classes_,
        probabilidades,
    ):
        print(
            f"{clase}: "
            f"{probabilidad:.4f}"
        )


if __name__ == "__main__":
    main()