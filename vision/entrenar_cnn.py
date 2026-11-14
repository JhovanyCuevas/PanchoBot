
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Rutas
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset_dividido"
SALIDA_DIR = BASE_DIR / "resultados"
SALIDA_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

# Cargar imágenes
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Categorías:", class_names)

# Optimizar carga de datos
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Aumento de datos
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Modelo ResNet50
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# Compilar
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Control del entrenamiento
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.00001
    )
]

# Entrenar
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Guardar modelo
model.save(SALIDA_DIR / "modelo_resnet50.keras")

# Evaluación con datos de prueba
loss, accuracy = model.evaluate(test_ds)

print("\nRESULTADOS DE EVALUACIÓN")
print("Pérdida:", round(loss, 4))
print("Accuracy:", round(accuracy, 4))

# Predicciones
y_real = np.concatenate([y.numpy() for x, y in test_ds])
predicciones = model.predict(test_ds)
y_pred = np.argmax(predicciones, axis=1)

print("\nREPORTE DE CLASIFICACIÓN")
print(classification_report(
    y_real,
    y_pred,
    target_names=class_names,
    zero_division=0
))

# Matriz de confusión
matriz = confusion_matrix(y_real, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues"
)
plt.xlabel("Predicción")
plt.ylabel("Categoría real")
plt.title("Matriz de confusión - PanchoBot")
plt.tight_layout()
plt.savefig(SALIDA_DIR / "matriz_confusion.png")
plt.show()

# Gráfica de entrenamiento
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Desempeño del modelo")
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA_DIR / "desempeno_modelo.png")
plt.show()

print("\nEntrenamiento finalizado.")
