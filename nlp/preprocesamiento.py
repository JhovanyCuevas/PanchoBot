
import re
import unicodedata
import pandas as pd
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Configuración de herramientas
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("spanish")
stop_words = set(stopwords.words("spanish"))

# Conservamos las negaciones porque pueden cambiar el significado
stop_words.discard("no")
stop_words.discard("ni")


# 1. Normalización del texto
def normalizar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


# 2. Tokenización
def tokenizar_texto(texto):
    return tokenizer.tokenize(texto)


# 3. Eliminación de stopwords
def eliminar_stopwords(tokens):
    return [palabra for palabra in tokens
            if palabra not in stop_words]


# 4. Stemming
def aplicar_stemming(tokens):
    return [stemmer.stem(palabra) for palabra in tokens]


# Procesamiento completo
def procesar_texto(texto):
    texto_normalizado = normalizar_texto(texto)
    tokens = tokenizar_texto(texto_normalizado)
    tokens_limpios = eliminar_stopwords(tokens)
    palabras_raiz = aplicar_stemming(tokens_limpios)

    return {
        "normalizado": texto_normalizado,
        "tokens": tokens,
        "sin_stopwords": tokens_limpios,
        "stemming": palabras_raiz
    }


# Leer el conjunto de datos
datos = pd.read_csv("nlp/dataset_intenciones_panchoBot_300_V2.csv")

# Aplicar el procesamiento
datos["texto_normalizado"] = datos["texto"].apply(normalizar_texto)

datos["tokens"] = datos["texto_normalizado"].apply(tokenizar_texto)

datos["tokens_limpios"] = datos["tokens"].apply(eliminar_stopwords)

datos["stemming"] = datos["tokens_limpios"].apply(aplicar_stemming)

# Guardar los resultados
datos.to_csv("nlp/dataset_preprocesado.csv",
             index=False, encoding="utf-8-sig")

# Mostrar algunos ejemplos
print("\nResultados del preprocesamiento:\n")

for _, fila in datos.head(10).iterrows():
    print("Texto original:", fila["texto"])
    print("Normalizado:", fila["texto_normalizado"])
    print("Tokens:", fila["tokens"])
    print("Sin stopwords:", fila["tokens_limpios"])
    print("Stemming:", fila["stemming"])
    print("-" * 50)

print("\nPreprocesamiento terminado.")
print("Total de registros procesados:", len(datos))
