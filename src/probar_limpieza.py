from src.preprocess import limpiar_texto


def main() -> None:
    textos = [
        "Apple reports record revenue and strong iPhone sales.",
        "Apple faces a major antitrust investigation https://example.com",
        "@investor Apple stock falls after weak guidance #AAPL",
    ]

    for texto in textos:
        limpio = limpiar_texto(texto)

        print("\nTexto original:")
        print(texto)

        print("Texto limpio:")
        print(limpio)


if __name__ == "__main__":
    main()