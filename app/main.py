import sys
from PyQt5.QtWidgets import QApplication
from model.image_model import ImageModel
from model.database_model import DatabaseModel
from model.prediction_model import PredictionModel
from view.main_window import MainWindow
from controller.main_controller import MainController

def main():
    """
    Punto de entrada de la aplicación.
    """
    # Inicializar la aplicación PyQt
    app = QApplication(sys.argv)

    # Inicializar los modelos
    image_model = ImageModel()
    database_model = DatabaseModel()
    prediction_model = PredictionModel(database_model)  # Pasar database_model a PredictionModel

    # Inicializar la vista
    vista = MainWindow()

    # Inicializar el controlador
    controlador = MainController(
        modelo={
            "image": image_model,
            "database": database_model,
            "prediction": prediction_model
        },
        vista=vista
    )

    # Mostrar la ventana principal
    vista.show()

    # Ejecutar la aplicación
    sys.exit(app.exec_())

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
# │   ├── main_controller.py        # Controlador principal (conexión entre vista y modelo)
# │   ├── image_controller.py       # Controlador para manejar la lógica de imágenes 
# │   ├── database_controller.py    # Controlador para manejar la base de datos de imagenes 
# │   └── prediction_controller.py  # Controlador para manejar la predicción
# │
# ├── resource/
# │   ├── dataset/                # Carpeta con imágenes para la base de datos
# │   │   ├── internet/
# │   │   │   ├── arandelas
# │   │   │   ├── clavos
# │   │   │   ├── tornillos
# │   │   │   └── tuercas
# │   │   └── photos/
# │   │       ├── arandelas
# │   │       ├── clavos
# │   │       ├── tornillos
# │   │       └── tuercas
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   ├── img3.jpg
# │   └── img4.jpg
# │
# └── main.py                     # Punto de entrada de la aplicación