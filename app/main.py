import sys
import os
import traceback
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from model.image_model import ImageModel
from model.database_model import DatabaseModel
from model.prediction_model import PredictionModel
from view.main_window import MainWindow
from controller.main_controller import MainController

def aplicar_tema_oscuro(app):
    """
    Aplica un tema personalizado basado en la paleta de colores del ícono (Azul Noche/Cian/Púrpura).
    """
    app.setStyle("Fusion")
    
    # --- PALETA DE COLORES EXTRAÍDA DEL ÍCONO ---
    COLOR_FONDO       = QColor("#181a26")  # Azul noche profundo (Base)
    COLOR_PANELES     = QColor("#222533")  # Azul grisáceo (Inputs, Listas)
    COLOR_TEXTO       = QColor("#e0e0e0")  # Blanco suave
    COLOR_ACENTO      = QColor("#2bd9fe")  # Cian eléctrico (Ojos del personaje) - Para selecciones
    COLOR_SECUNDARIO  = QColor("#7d54ae")  # Púrpura (Ropa del personaje) - Para detalles
    COLOR_BOTON       = QColor("#2f3447")  # Fondo de botón reposo
    COLOR_HOVER       = QColor("#3d4259")  # Fondo de botón al pasar mouse
    
    dark_palette = QPalette()
    
    # Configuración base de la paleta Qt
    dark_palette.setColor(QPalette.Window, COLOR_FONDO)
    dark_palette.setColor(QPalette.WindowText, COLOR_TEXTO)
    dark_palette.setColor(QPalette.Base, COLOR_PANELES)
    dark_palette.setColor(QPalette.AlternateBase, COLOR_FONDO)
    dark_palette.setColor(QPalette.ToolTipBase, COLOR_ACENTO)
    dark_palette.setColor(QPalette.ToolTipText, QColor("#000000")) # Texto negro sobre cian
    dark_palette.setColor(QPalette.Text, COLOR_TEXTO)
    
    # Botones
    dark_palette.setColor(QPalette.Button, COLOR_BOTON)
    dark_palette.setColor(QPalette.ButtonText, COLOR_TEXTO)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    
    # Enlaces y Resaltados
    dark_palette.setColor(QPalette.Link, COLOR_ACENTO)
    dark_palette.setColor(QPalette.Highlight, COLOR_ACENTO)
    dark_palette.setColor(QPalette.HighlightedText, QColor("#181a26")) # Texto oscuro sobre selección cian
    
    app.setPalette(dark_palette)
    
    # --- HOJA DE ESTILOS (CSS) REFINADA ---
    app.setStyleSheet(f"""
        /* Tooltips estilizados */
        QToolTip {{ 
            color: #000000; 
            background-color: {COLOR_ACENTO.name()}; 
            border: 1px solid white; 
            font-weight: bold;
        }}
        
        /* Cajas de Texto (Logs y STRIPS) */
        QTextEdit {{
            background-color: {COLOR_PANELES.name()};
            color: {COLOR_ACENTO.name()};  /* Texto Cian estilo Sci-Fi */
            border: 1px solid #444;
            font-family: Consolas, Monospace;
            font-size: 12px;
        }}

        /* Botones Modernos */
        QPushButton {{
            background-color: {COLOR_BOTON.name()};
            border: 1px solid #4a4a4a;
            padding: 6px 12px;
            border-radius: 6px;
            color: white;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {COLOR_HOVER.name()};
            border: 1px solid {COLOR_ACENTO.name()}; /* Borde cian al pasar mouse */
        }}
        QPushButton:pressed {{
            background-color: {COLOR_SECUNDARIO.name()}; /* Se pone Púrpura al hacer click */
            border: 1px solid {COLOR_ACENTO.name()};
        }}

        /* Pestañas (Tabs) */
        QTabBar::tab {{
            background: {COLOR_FONDO.name()};
            color: #888;
            padding: 8px 25px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }}
        QTabBar::tab:selected {{
            background: {COLOR_PANELES.name()};
            color: {COLOR_ACENTO.name()};
            border-bottom: 2px solid {COLOR_ACENTO.name()}; /* Línea cian debajo */
        }}
        QTabWidget::pane {{ 
            border: 1px solid #444; 
            background-color: {COLOR_PANELES.name()};
        }}
        
        /* Listas (STRIPS) */
        QListWidget {{
            background-color: {COLOR_PANELES.name()};
            color: white;
            border: 1px solid #444;
            border-radius: 4px;
        }}
        QListWidget::item:selected {{
            background-color: {COLOR_ACENTO.name()};
            color: #000000;
        }}
        
        /* Barra de progreso o Scrollbars (Opcional, para completar el look) */
        QScrollBar:vertical {{
            border: none;
            background: {COLOR_FONDO.name()};
            width: 10px;
            margin: 0px 0px 0px 0px;
        }}
        QScrollBar::handle:vertical {{
            background: #444;
            min-height: 20px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {COLOR_SECUNDARIO.name()};
        }}
    """)

def main():
    try:
        app = QApplication(sys.argv)
        
        # --- APLICAR TEMA OSCURO ---
        aplicar_tema_oscuro(app)
        
        # --- Inicialización de Modelos ---
        image_model = ImageModel()
        database_model = DatabaseModel()
        prediction_model = PredictionModel(database_model)

        modelos = {
            "image": image_model,
            "database": database_model,
            "prediction": prediction_model
        }

        # --- Inicialización de Vista y Controlador ---
        vista = MainWindow()
        controlador = MainController(modelo=modelos, vista=vista)

        vista.show()
        sys.exit(app.exec())

    except Exception as e:
        print("Error fatal al iniciar la aplicación:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()