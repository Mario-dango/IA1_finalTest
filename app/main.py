import sys                                 # Importa el módulo sys para gestionar argumentos y la salida del programa
from PyQt5.QtWidgets import QApplication    # Importa QApplication, necesaria para inicializar la aplicación PyQt
from model.image_model import ImageModel      # Importa el modelo para procesamiento de imágenes
from model.database_model import DatabaseModel  # Importa el modelo que gestiona la base de datos de imágenes
from model.prediction_model import PredictionModel  # Importa el modelo para clasificación (KNN y K-means)
from view.main_window import MainWindow       # Importa la ventana principal (vista)
from controller.main_controller import MainController  # Importa el controlador principal

def main():
    """
    Función principal que inicializa y ejecuta la aplicación.
    """
    # Inicializa la aplicación PyQt
    app = QApplication(sys.argv)
    
    # Inicializa los modelos necesarios
    image_model = ImageModel()
    database_model = DatabaseModel()
    # Se pasa el modelo de base de datos a PredictionModel para la normalización, etc.
    prediction_model = PredictionModel(database_model)

    # Inicializa la vista (ventana principal)
    vista = MainWindow()

    # Inicializa el controlador, pasando un diccionario con los modelos y la vista
    controlador = MainController(
        modelo={
            "image": image_model,
            "database": database_model,
            "prediction": prediction_model
        },
        vista=vista
    )

    # Muestra la ventana principal
    vista.show()

    # Ejecuta el ciclo de eventos de la aplicación y finaliza cuando se cierra la ventana
    sys.exit(app.exec_())

# Punto de entrada: si este script se ejecuta directamente, se llama a main()
if __name__ == "__main__":
    main()

    
# app/
# │
# ├── model/
# │   ├── __init__.py
# │   ├── image_model.py          # Maneja la lógica de la imagen (carga, filtrado, contornos, DFT de la imagen a seleccionar)
# │   ├── database_model.py       # Maneja la base de datos (carga todas las imagenes de una carpeta con sub-carpetas, (aplica filtrado, contornos, DFT en todas las imagenes que recive de database_model.py))
# │   └── prediction_model.py     # Maneja la predicción (KNN, K-means que tienen que ser desarrollados sin usar librerías externas y exportar una imagen 3d de los agrupamientos según sus caracteristicas)
# │
# ├── view/
# │   ├── __init__.py
# │   ├── main_window.py          # Ventana principal con todos los widgets
# │   ├── image_view.py           # Muestra las imágenes (original, filtrada, KNN, K-means)
# │   └── text_view.py            # Muestra la información procesada
# │
# ├── controller/
# │   ├── __init__.py
# │   └── main_controller.py        # Controlador principal (conexión entre vista y modelo)
# │
# ├── resource/
# │   ├── dataset/                # Carpeta con imágenes para la base de datos
# │   │   ├── arandelas
# │   │   ├── clavos
# │   │   ├── tornillos
# │   │   └── tuercas
# │
# └── main.py                     # Punto de entrada de la aplicación