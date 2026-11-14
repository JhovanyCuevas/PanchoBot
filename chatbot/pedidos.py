
def registrar_pedido():
    print("\n--- Registro de pedido ---")
    print("Escribe 'cancelar' en cualquier momento para detener el registro.")

    def solicitar_dato(pregunta):
        dato = input(pregunta).strip()

        if dato.lower() in ["cancelar", "salir", "atras", "regresar", "anterior"]:
            return None

        return dato

    nombre = solicitar_dato("Nombre del cliente: ")
    if nombre is None:
        return False

    producto = solicitar_dato("¿Qué pedido quiere realizar?: ")
    if producto is None:
        return False

    cantidad = solicitar_dato("¿Qué cantidad necesita?: ")
    if cantidad is None:
        return False

    fecha = solicitar_dato("¿Para qué fecha?: ")
    if fecha is None:
        return False

    sucursal = solicitar_dato("¿En qué sucursal?: ")
    if sucursal is None:
        return False

    telefono = solicitar_dato("Teléfono de contacto (opcional): ")
    if telefono is None:
        return False

    print("\n--- Resumen del pedido ---")
    print(f"Cliente: {nombre}")
    print(f"Producto: {producto}")
    print(f"Cantidad: {cantidad}")
    print(f"Fecha: {fecha}")
    print(f"Sucursal: {sucursal}")
    print(f"Teléfono: {telefono or 'No proporcionado'}")

    print(
        "\nTu solicitud ha sido registrada en esta sesión."
        "\nLa panadería deberá confirmar el precio y la disponibilidad."
        "\nNos pondremos en contacto contigo lo antes posible."
    )

    return True
