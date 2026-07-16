import shutil
from pathlib import Path

import nltk


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESTINATION = PROJECT_ROOT / "nltk_data" / "corpora" / "stopwords"


def main() -> None:
    try:
        source_pointer = nltk.data.find("corpora/stopwords")
        source = Path(str(source_pointer))
    except LookupError as error:
        raise RuntimeError(
            "NLTK no ha encontrado las stopwords instaladas."
        ) from error

    print(f"Origen: {source}")
    print(f"Destino: {DESTINATION}")

    DESTINATION.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)

    shutil.copytree(
        source,
        DESTINATION,
    )

    print("Stopwords copiadas correctamente.")


if __name__ == "__main__":
    main()