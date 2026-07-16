import json
import os
from pathlib import Path

from src import score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY = PROJECT_ROOT / "models"
REQUEST_PATH = PROJECT_ROOT / "requests" / "sample-request.json"


def main() -> None:
    if not MODEL_DIRECTORY.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de modelos: {MODEL_DIRECTORY}"
        )

    if not REQUEST_PATH.exists():
        raise FileNotFoundError(
            f"No existe el fichero de petición: {REQUEST_PATH}"
        )

    # Simula la variable que Azure crea al desplegar el modelo.
    os.environ["AZUREML_MODEL_DIR"] = str(MODEL_DIRECTORY)

    # Carga el modelo utilizando la función del endpoint.
    score.init()

    # Lee el JSON de prueba.
    request_content = REQUEST_PATH.read_text(
        encoding="utf-8"
    )

    # Ejecuta una predicción.
    response = score.run(request_content)

    print("\nRESPUESTA DEL MODELO:\n")
    print(
        json.dumps(
            response,
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()