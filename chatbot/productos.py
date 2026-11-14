
# Productos y sucursales de la panadería

productos = {
    "bolillo": {
        "precio": 2.50,
        "tipo": "pieza"
    },
    "pan de dulce": {
        "precio": 10.00,
        "tipo": "pieza"
    },
    "feite": {
        "precio": 12.00,
        "tipo": "pieza"
    },
    "pasteles": {
        "precio": None,
        "tipo": "pedido especial"
    },
    "pan de temporada": {
        "precio": None,
        "tipo": "pedido especial"
    }
}

sucursales = [
    "Victoria",
    "San Juditas",
    "Conchita"
]


def mostrar_productos():
    print("\nProductos disponibles:")

    for nombre in productos:
        print(f"- {nombre.capitalize()}")


def consultar_precio(nombre):
    for producto, datos in productos.items():
        if producto in nombre:
            if datos["precio"] is not None:
                return (
                    f"El precio de {producto} es de "
                    f"${datos['precio']:.2f} por {datos['tipo']}."
                )
            else:
                return (
                    f"{producto.capitalize()} se prepara bajo pedido. "
                    "El precio debe confirmarse con la panadería."
                )

    return "No encontré ese producto. ¿Puedes especificar cuál buscas?"
