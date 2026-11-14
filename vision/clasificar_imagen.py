
from pathlib import Path
import tensorflow as tf
import numpy as np

# Rutas
BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / "resultados" / "modelo_resnet50.keras"

# Cargar modelo entrenado
modelo = tf.keras.models.load_model(MODELO_PATH)

# Orden alfabético utilizado por Keras al cargar las carpetas
CLASES = ["bolillo", "feite", "pan_dulce", "pasteles"]


def clasificar_imagen(ruta_imagen):
    ruta_imagen = Path(ruta_imagen)

    if not ruta_imagen.exists():
        return "No se encontró la imagen."

    try:
        imagen = tf.keras.utils.load_img(
            ruta_imagen,
            target_size=(224, 224)
        )

        imagen_array = tf.keras.utils.img_to_array(imagen)
        imagen_array = np.expand_dims(imagen_array, axis=0)

        predicciones = modelo.predict(imagen_array, verbose=0)

        indice = np.argmax(predicciones[0])
        categoria = CLASES[indice]
        confianza = float(predicciones[0][indice]) * 100

        return {
            "categoria": categoria,
            "confianza": round(confianza, 2),
            "probabilidades": {
                 CLASES[i]: round(float(predicciones[0][i]) * 100, 2)
                for i in range(len(CLASES))
            }
        }

    except Exception as error:
        return f"Error al procesar la imagen: {error}"


if __name__ == "__main__":
    ruta = input("Introduce la ruta de la imagen: ")
    resultado = clasificar_imagen(ruta)

    print("\nResultado:")
    print(resultado)
