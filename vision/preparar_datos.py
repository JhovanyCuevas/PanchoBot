
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Carpetas
RUTA_ORIGEN = Path("vision/dataset")
RUTA_DESTINO = Path("vision/dataset_dividido")

CATEGORIAS = ["bolillo", "pan_dulce", "feite", "pasteles"]

EXTENSIONES = [".jpg", ".jpeg", ".png", ".webp"]

# Crear carpetas de destino
for conjunto in ["train", "val", "test"]:
    for categoria in CATEGORIAS:
        (RUTA_DESTINO / conjunto / categoria).mkdir(
            parents=True, exist_ok=True
        )

print("DIVISIÓN DEL DATASET")
print("--------------------")

for categoria in CATEGORIAS:

    carpeta = RUTA_ORIGEN / categoria

    imagenes = [
        archivo for archivo in carpeta.iterdir()
        if archivo.is_file()
        and archivo.suffix.lower() in EXTENSIONES
    ]

    # Primera división: 70% entrenamiento y 30% restante
    entrenamiento, restante = train_test_split(
        imagenes,
        test_size=0.30,
        random_state=42
    )

    # Segunda división: 15% validación y 15% prueba
    validacion, prueba = train_test_split(
        restante,
        test_size=0.50,
        random_state=42
    )

    conjuntos = {
        "train": entrenamiento,
        "val": validacion,
        "test": prueba
    }

    for nombre, archivos in conjuntos.items():

        for archivo in archivos:
            destino = RUTA_DESTINO / nombre / categoria / archivo.name
            shutil.copy2(archivo, destino)

        print(f"{categoria} - {nombre}: {len(archivos)} imágenes")

print("\nDivisión del dataset terminada.")
