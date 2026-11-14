# PanchoBot - Chatbot de panadería


## 1. Descripción del proyecto

PanchoBot es un chatbot desarrollado en Python como parte de un proyecto escolar. 
Su objetivo es ayudar a una panadería a responder preguntas frecuentes de los clientes sobre productos, 
precios, horarios, sucursales y pedidos.

El chatbot utiliza reglas y palabras clave para identificar las consultas y generar respuestas. 
Esta versión funciona desde la terminal y no está conectada a alguna aplicación de mensajeria
ni a una base de datos externa.



## 2. Archivos del proyecto

- **main.py:** contiene la lógica principal del chatbot y las reglas para responder a los mensajes del usuario.
- **productos.py:** contiene el catálogo de productos, sus precios y las funciones para consultar la información.
- **pedidos.py:** permite recopilar los datos de un pedido y mostrar un resumen de la solicitud.
- **evidencias/:** carpeta con diez capturas de pantalla de las pruebas realizadas.



## 3. Requisitos

- Python 3.10 o superior.
- Biblioteca NLTK.



## 4. Instalación

1. Instalar Python 3.10 o superior.
2. Abrir una terminal en la carpeta del proyecto.
3. Instalar la biblioteca necesaria ejecutando el siguiente comando:

pip install nltk




## 5. Ejecución

Desde la terminal, ejecutar:

python main.py

Una vez iniciado el programa, se mostrará un mensaje de bienvenida y el usuario podrá escribir preguntas 
sobre los servicios de la panadería.

Para finalizar la conversación, escribir:

salir



## 6. Pruebas

En la carpeta evidencias se incluyen diez capturas de pantalla de las pruebas realizadas para comprobar 
las respuestas del chatbot ante diferentes consultas.




## 7. Limitaciones

PanchoBot utiliza reglas y palabras clave previamente definidas. Por esta razón, algunas preguntas pueden 
no ser reconocidas si no coinciden con las condiciones programadas. Los pedidos se registran únicamente 
durante la ejecución del programa y requieren confirmación por parte de la panadería.