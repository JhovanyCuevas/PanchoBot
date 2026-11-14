
import re
import unicodedata
from pathlib import Path
import joblib

from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from vision.clasificar_imagen import clasificar_imagen

import chatbot.productos as productos
import chatbot.pedidos as pedidos


# ==========================================
# 1. CARGAR MODELO Y VECTORIZADOR
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

modelo_svm = joblib.load(BASE_DIR / "nlp" / "modelo_svm.pkl")
vectorizador = joblib.load(BASE_DIR / "nlp" / "vectorizador_tfidf.pkl")


# ==========================================
# 2. CONFIGURAR PROCESAMIENTO DE TEXTO
# ==========================================

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("spanish")

stop_words = set(stopwords.words("spanish"))
stop_words.discard("no")
stop_words.discard("ni")

producto_seleccionado = None
esperando_producto = False


def normalizar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# Procesar el mensaje igual que durante el entrenamiento
def procesar_mensaje(texto):
    texto = normalizar_texto(texto)
    tokens = tokenizer.tokenize(texto)

    tokens_limpios = [
        palabra for palabra in tokens
        if palabra not in stop_words
    ]

    palabras_raiz = [
        stemmer.stem(palabra)
        for palabra in tokens_limpios
    ]

    return " ".join(palabras_raiz)


# ==========================================
# 3. IDENTIFICAR INTENCIÓN CON SVM
# ==========================================

def identificar_intencion(mensaje):

    texto_procesado = procesar_mensaje(mensaje)

    if not texto_procesado:
        return None

    texto_tfidf = vectorizador.transform([texto_procesado])

    intencion = modelo_svm.predict(texto_tfidf)[0]

    return intencion


# ==========================================
# 4. IDENTIFICAR PRODUCTOS
# ==========================================

def identificar_producto(texto):

    texto = normalizar_texto(texto)

    productos_conocidos = [
        "pan de dulce",
        "pan dulce",
        "bolillo",
        "feite",
        "pasteles",
        "pastel",
        "pan de temporada"
    ]

    for producto in productos_conocidos:

        if producto in texto:

            if producto == "pan dulce":
                return "pan de dulce"

            if producto == "pastel":
                return "pasteles"

            return producto

    return None




def identificar_producto_imagen():
    global producto_seleccionado, esperando_producto

    print("\nPanchoBot: Comparte la ruta de la imagen del producto.")
    ruta = input("Ruta de la imagen: ").strip().strip('"')

    resultado = clasificar_imagen(ruta)

    if not isinstance(resultado, dict):
        return resultado

    categoria = resultado["categoria"]
    confianza = resultado["confianza"]

    nombres = {
        "bolillo": "bolillo",
        "feite": "feite",
        "pan_dulce": "pan de dulce",
        "pasteles": "pasteles"
    }

    if confianza < 60:
        return (
            "No pude identificar el producto con suficiente seguridad.\n"
            f"Resultados: {resultado['probabilidades']}\n"
            "Intenta con otra fotografía."
        )

    # Guardar el producto identificado
    producto_seleccionado = nombres.get(categoria, categoria)
    esperando_producto = False

    return (
        f"Identifiqué el producto como: {producto_seleccionado}.\n"
        f"Confianza del modelo: {confianza}%.\n"
        "¿Quieres consultar su precio o realizar un pedido?"
    )






# ==========================================
# 5. GENERAR RESPUESTAS
# ==========================================

def responder(mensaje):

    global producto_seleccionado, esperando_producto

    texto = normalizar_texto(mensaje)


    # IDENTIFICAR PRODUCTO MEDIANTE IMAGEN

    solicitudes_imagen = [
        "identificar imagen", "identificar una imagen",
        "identificar producto", "identificar un producto",
        "identificar fotografia", "identificar una fotografia",
        "identificar foto", "identificar una foto",

        "reconocer imagen", "reconocer una imagen",
        "reconocer producto", "reconocer un producto",
        "reconocer fotografia", "reconocer una fotografia",
        "reconocer foto", "reconocer una foto",

        "analizar imagen", "analizar una imagen",
        "analizar producto", "analizar un producto",
        "analizar fotografia", "analizar una fotografia",
        "analizar foto", "analizar una foto",

        "clasificar imagen", "clasificar una imagen",
        "clasificar producto", "clasificar un producto",
        "clasificar fotografia", "clasificar una fotografia",
        "clasificar foto", "clasificar una foto",

        "que producto es",
        "que pan es",
        "que es esto",
        "hacen este", "hacen esto",
        "que tipo de pan es"
    ]

    if any(frase in texto for frase in solicitudes_imagen):
        return identificar_producto_imagen()


    # Si se esperaba el nombre de un producto
    if esperando_producto:

        producto = identificar_producto(texto)

        if producto:
            producto_seleccionado = producto
            esperando_producto = False
            return productos.consultar_precio(producto)

        return "No identifiqué el producto. ¿Puedes escribir su nombre?"


    # Identificar intención mediante SVM
    intencion = identificar_intencion(mensaje)

    if intencion is None:
        return "No entendí tu consulta. ¿Puedes escribirla nuevamente?"


    # CONSULTAR PRECIOS
    if intencion == "consultar_precios":

        producto = identificar_producto(texto)

        if producto:
            producto_seleccionado = producto
            return productos.consultar_precio(producto)

        elif producto_seleccionado:
            return productos.consultar_precio(producto_seleccionado)

        else:
            esperando_producto = True
            return "¿De qué producto quieres conocer el precio?"


    # MAYOREO
    elif intencion == "mayoreo":

        producto_seleccionado = None
        esperando_producto = False

        return (
            "Podemos atender solicitudes de pan para reventa. "
            "Los precios dependen del producto y la cantidad solicitada. "
            "¿Qué productos y qué cantidad te interesan?"
        )


    # CONSULTAR PRODUCTOS
    elif intencion == "consultar_productos":

        producto = identificar_producto(texto)

        if producto:
            producto_seleccionado = producto

            return (
                f"Seleccionaste {producto}. "
                "¿Quieres consultar su precio o realizar un pedido?"
            )

        productos.mostrar_productos()

        return "¿Sobre cuál producto te gustaría recibir información?"


    # PASTELES
    elif intencion == "pasteles":

        return (
            "Los pasteles se preparan bajo pedido. "
            "Escribe 'hacer pedido' para registrar tu solicitud."
        )


    # HORARIOS
    elif intencion == "horarios":

        return (
            "Contamos con dos horarios diferentes:"
            "\nSucursal Victoria: Lunes a domingo, de 7:00 a.m. a 10:00 p.m."
            "\nSucursal San Juditas y Conchita: Lunes a domingo, "
            "de 7:00 a.m. a 11:00 a.m. y de 4:00 p.m. a 9:00 p.m."
        )


    # SUCURSALES
    elif intencion == "sucursales":

        return (
            "Contamos con tres sucursales:"
            "\nVictoria: C. Nicolas Bravo #74, Barrio San Juan."
            "\nSan Juditas: Camino Viejo a Melchor Ocampo #53, "
            "Barrio San Miguel Otlica."
            "\nConchita: Cerrada Nicolas Bravo S/N, Barrio San Juan."
        )


    # REGISTRAR PEDIDO
    elif intencion == "registrar_pedido":

        pedido_registrado = pedidos.registrar_pedido()

        if not pedido_registrado:
            return "El registro del pedido fue cancelado."

        return (
        "Gracias por tu solicitud. "
        "¿Te puedo ayudar en algo más?"
    )


    # AYUDA
    elif intencion == "ayuda":

        return (
            "Puedo ayudarte con lo siguiente:"
            "\n- Consultar productos y precios."
            "\n- Conocer las sucursales y sus horarios."
            "\n- Solicitar pasteles y pedidos especiales."
            "\n- Consultar información para compras de mayoreo."
            "\n¿Qué información necesitas?"
        )


    # SALUDO
    elif intencion == "saludo":

        return "¡Hola! Soy PanchoBot. ¿En qué puedo ayudarte?"


    # DESPEDIDA
    elif intencion == "despedida":

        return "¡Gracias por comunicarte con nosotros!"


    # RESPUESTA POR DEFECTO
    else:

        return (
            "No entendí tu consulta. Puedes preguntar por productos, "
            "precios, ubicaciones o escribir 'hacer pedido'."
        )


# ==========================================
# 6. INICIAR CONVERSACIÓN
# ==========================================

def iniciar_chat():

    print("======================================")
    print("              PANCHOBOT")
    print("     Asistente virtual de panadería")
    print(" ")
    print("  ¿Cómo puedo ayudarte el día de hoy?")
    print("Puedes preguntar por productos, precios,")
    print("  ubicaciones o realizar un pedido.")
    print("Escribe 'salir' para terminar.")
    print("======================================")

    while True:

        mensaje = input("\nTú: ")

        if normalizar_texto(mensaje) == "salir":

            print("PanchoBot: ¡Gracias por comunicarte con nosotros!")
            break

        respuesta = responder(mensaje)

        print(f"\nPanchoBot: {respuesta}")


if __name__ == "__main__":
    iniciar_chat()
