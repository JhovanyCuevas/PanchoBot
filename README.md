# PanchoBot

**Chatbot inteligente para una panadería mediante procesamiento de lenguaje natural y visión por computadora.**

## Descripción

PanchoBot es un asistente conversacional desarrollado en Python para apoyar la atención a clientes de una panadería con tres sucursales: Victoria, San Juditas y Conchita.

El sistema permite consultar productos, precios, horarios y sucursales, así como registrar pedidos. Además, incorpora modelos de inteligencia artificial para identificar intenciones en mensajes de texto y reconocer productos de panadería a partir de imágenes.

## Objetivo

Desarrollar una primera versión multimodal de un chatbot que facilite la atención de consultas frecuentes y apoye la identificación de productos mediante imágenes.

## Funcionalidades

- Identificación de intenciones mediante procesamiento de lenguaje natural.
- Normalización, tokenización, eliminación de palabras vacías y stemming.
- Representación de texto mediante TF-IDF.
- Clasificación de intenciones mediante Support Vector Machine (SVM).
- Consulta de productos y precios.
- Información de sucursales y horarios.
- Registro de pedidos.
- Clasificación de imágenes de productos mediante ResNet50.
- Integración del componente conversacional y el componente visual.

## Tecnologías utilizadas

- Python
- NLTK
- Scikit-learn
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Pillow
- Joblib

## Estructura del proyecto

```text
PanchoBot/
│
├── chatbot/
│   ├── main.py
│   ├── productos.py
│   └── pedidos.py
│
├── nlp/
│   ├── dataset_intenciones_panchoBot_300_V2.csv
│   ├── dataset_preprocesado.csv
│   ├── preprocesamiento.py
│   ├── entrenamiento.py
│   ├── modelo_svm.pkl
│   └── vectorizador_tfidf.pkl
│
├── vision/
│   ├── entrenar_cnn.py
│   ├── preparar_datos.py
│   ├── clasificar_imagen.py
│   └── resultados/
│       ├── modelo_resnet50.keras
│       ├── matriz_confusion.png
│       └── desempeno_modelo.png
│
├── evidencias/
├── .gitignore
└── README.md
```

## Modelos implementados

### Procesamiento de lenguaje natural

Se construyó un conjunto de datos de 300 ejemplos distribuidos en 10 intenciones. Se aplicó preprocesamiento de texto y representación TF-IDF para entrenar un clasificador SVM.

**Resultados obtenidos:**
- Accuracy: 78.33 %.
- F1-score macro: 0.78.

### Visión por computadora

Se implementó una red neuronal convolucional utilizando transferencia de aprendizaje con ResNet50 y pesos preentrenados de ImageNet.

El clasificador reconoce cuatro categorías:

- Bolillo.
- Pan dulce.
- Feite.
- Pasteles.

**Resultados obtenidos:**
- Accuracy en prueba: 89.09 %.
- F1-score por categoría: entre 0.76 y 1.00.

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd PanchoBot
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

Activar en Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install nltk scikit-learn pandas numpy joblib tensorflow matplotlib pillow
```

Si NLTK no encuentra las palabras vacías en español, ejecutar:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 4. Ejecutar PanchoBot

Desde la carpeta raíz del proyecto:

```bash
python -m chatbot.main
```

El programa iniciará una conversación en la terminal.

## Clasificación de imágenes

El componente visual utiliza el modelo entrenado `modelo_resnet50.keras`, incluido en el repositorio.

Para utilizarlo se necesita proporcionar la ruta de una imagen compatible cuando PanchoBot solicite identificar un producto.

## Dataset visual

Las imágenes utilizadas para entrenar y evaluar el clasificador se organizaron en cuatro categorías y se dividieron en conjuntos de entrenamiento, validación y prueba.

Por su tamaño, las carpetas con imágenes no se incluyen en este repositorio.

Para volver a entrenar el modelo es necesario disponer del dataset y organizarlo de acuerdo con la estructura esperada por `preparar_datos.py` y `entrenar_cnn.py`.

## Limitaciones

- El reconocimiento de intenciones depende de los ejemplos utilizados durante el entrenamiento.
- El clasificador visual puede presentar confusiones entre productos con características similares.
- El sistema funciona desde una terminal y no cuenta con integración con plataformas externas de mensajería.
- La información de productos, precios y sucursales debe mantenerse actualizada.

## Autor

Proyecto académico desarrollado como parte de la Maestría en Inteligencia Artificial.

**PanchoBot – 2026**