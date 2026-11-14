
import pandas as pd
import ast
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Cargar el conjunto de datos preprocesado
datos = pd.read_csv("nlp/dataset_preprocesado.csv")

# 2. Recuperar las palabras procesadas
datos["texto_procesado"] = datos["stemming"].apply(
    lambda x: " ".join(ast.literal_eval(x))
)

# 3. Separar preguntas e intenciones
X = datos["texto_procesado"]
y = datos["intencion"]

# 4. Dividir los datos: 80% entrenamiento y 20% prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Convertir los textos a valores TF-IDF
vectorizador = TfidfVectorizer(ngram_range=(1, 2))

X_entrenamiento_tfidf = vectorizador.fit_transform(X_entrenamiento)
X_prueba_tfidf = vectorizador.transform(X_prueba)

# 6. Crear y entrenar el clasificador Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_entrenamiento_tfidf, y_entrenamiento)

# 7. Realizar predicciones
predicciones = modelo.predict(X_prueba_tfidf)


# Analysis con SVM
# 9. Entrenar el clasificador SVM
modelo_svm = LinearSVC(random_state=42)
modelo_svm.fit(X_entrenamiento_tfidf, y_entrenamiento)

# 10. Realizar predicciones con SVM
predicciones_svm = modelo_svm.predict(X_prueba_tfidf)

# 11. Evaluar ambos modelos
precision_nb = accuracy_score(y_prueba, predicciones)
precision_svm = accuracy_score(y_prueba, predicciones_svm)

print("\nCOMPARACIÓN DE CLASIFICADORES")
print("--------------------------------")
print("Naive Bayes:", round(precision_nb * 100, 2), "%")
print("SVM:", round(precision_svm * 100, 2), "%")

print("\nREPORTE DE CLASIFICACIÓN SVM")
print("--------------------------------")
print(classification_report(
    y_prueba, predicciones_svm, zero_division=0
))

print("\nMATRIZ DE CONFUSIÓN SVM")
print("--------------------------------")
print(confusion_matrix(y_prueba, predicciones_svm))


# 12. Identificar errores de clasificación de SVM
resultados_svm = pd.DataFrame({
    "pregunta": X_prueba.values,
    "intencion_real": y_prueba.values,
    "intencion_predicha": predicciones_svm
})

errores_svm = resultados_svm[
    resultados_svm["intencion_real"] !=
    resultados_svm["intencion_predicha"]
]

print("\nPREGUNTAS CLASIFICADAS INCORRECTAMENTE POR SVM")
print("----------------------------------------------")
print("Total de errores:", len(errores_svm))

for _, fila in errores_svm.iterrows():
    print("\nPregunta:", fila["pregunta"])
    print("Intención real:", fila["intencion_real"])
    print("Intención predicha:", fila["intencion_predicha"])




# Analysis con Naive Bayes
# 8. Evaluar el modelo
precision = accuracy_score(y_prueba, predicciones)

print("ENTRENAMIENTO DEL CLASIFICADOR")
print("--------------------------------")
print("Preguntas para entrenamiento:", len(X_entrenamiento))
print("Preguntas para prueba:", len(X_prueba))
print("Exactitud del modelo:", round(precision * 100, 2), "%")

print("\nReporte de clasificación:")
print(classification_report(y_prueba, predicciones, zero_division=0))

print("\nMatriz de confusión:")
print(confusion_matrix(y_prueba, predicciones))


# 9. Identificar preguntas clasificadas incorrectamente
resultados = pd.DataFrame({
    "pregunta": X_prueba.values,
    "intencion_real": y_prueba.values,
    "intencion_predicha": predicciones
})

errores = resultados[
    resultados["intencion_real"] != resultados["intencion_predicha"]
]

print("\nPREGUNTAS CLASIFICADAS INCORRECTAMENTE")
print("--------------------------------------")
print("Total de errores:", len(errores))

for _, fila in errores.iterrows():
    print("\nPregunta:", fila["pregunta"])
    print("Intención real:", fila["intencion_real"])
    print("Intención predicha:", fila["intencion_predicha"])



import joblib

# Guardar el modelo y el vectorizador
joblib.dump(modelo_svm, "nlp/modelo_svm.pkl")
joblib.dump(vectorizador, "nlp/vectorizador_tfidf.pkl")

print("\nModelo SVM y vectorizador guardados correctamente.")