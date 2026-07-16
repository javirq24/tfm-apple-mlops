from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_sentimiento.joblib"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo en: {MODEL_PATH}"
        )

    modelo = joblib.load(MODEL_PATH)

    print("Modelo cargado correctamente.")
    print("Tipo de objeto:", type(modelo))
    print("Pasos del pipeline:", list(modelo.named_steps.keys()))
    print("Clases:", modelo.classes_)

    textos = [
        "Apple reports record revenue and strong iPhone sales",
        "Apple faces a major antitrust investigation",
        "Apple has released a software update",
    ]

    predicciones = modelo.predict(textos)
    probabilidades = modelo.predict_proba(textos)

    for texto, prediccion, probabilidades_texto in zip(
        textos,
        predicciones,
        probabilidades,
    ):
        print("\nTexto:", texto)
        print("Predicción:", prediccion)

        for clase, probabilidad in zip(
            modelo.classes_,
            probabilidades_texto,
        ):
            print(f"Probabilidad {clase}: {probabilidad:.4f}")


if __name__ == "__main__":
    main()