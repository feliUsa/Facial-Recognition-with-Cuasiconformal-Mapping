from face_system import FaceSystem


def main():
    system = FaceSystem()

    while True:
        print("\n=== Sistema de Reconocimiento Facial ===")
        print("1) Registrar usuario")
        print("2) Login")
        print("q) Salir")
        choice = input("Selecciona una opción: ").strip().lower()

        if choice == "1":
            name = input("Nombre del usuario a registrar: ").strip()
            if not name:
                print("Nombre vacío, intenta de nuevo.")
                continue
            system.register_user(name)

        elif choice == "2":
            system.login()

        elif choice in ("q", "quit", "salir"):
            print("Saliendo.")
            break

        else:
            print("Opcion no válida. Intenta de nuevo.")


if __name__ == "__main__":
    main()
