from src.preprocess import limpiar_texto


def test_elimina_url():
    texto = (
        "Apple rises strongly "
        "https://example.com"
    )

    resultado = limpiar_texto(texto)

    assert "http" not in resultado


def test_elimina_mencion():
    texto = (
        "@usuario Apple reports "
        "strong results"
    )

    resultado = limpiar_texto(texto)

    assert "usuario" not in resultado


def test_devuelve_cadena():
    resultado = limpiar_texto(
        "Apple reports excellent earnings"
    )

    assert isinstance(
        resultado,
        str,
    )


def test_resultado_no_vacio():
    resultado = limpiar_texto(
        "Investors celebrate strong earnings"
    )

    assert len(resultado) > 0