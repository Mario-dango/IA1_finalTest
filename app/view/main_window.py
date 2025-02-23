from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit, QFileDialog, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from view.image_view import Grafico3DCanvas
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        """
        Inicializa la ventana principal de la aplicación.
        """
        super().__init__()
        self.setWindowTitle("Dashboard VisionAR")
        self.init_ui()

    def init_ui(self):
        """
        Configura la interfaz gráfica de la ventana principal.
        """
        # Widget central y layout principal
        widget_central = QWidget()
        layout_principal = QVBoxLayout(widget_central)
        self.setCentralWidget(widget_central)

        # Pestañas
        self.tabs = QTabWidget()
        layout_principal.addWidget(self.tabs)

        # Pestaña "Detección"
        self.tab_imagen = QWidget()
        self.tabs.addTab(self.tab_imagen, "Detección")
        self.layout_tab_imagen = QGridLayout(self.tab_imagen)
        self.layout_tab_imagen.setColumnStretch(0, 1)
        self.layout_tab_imagen.setColumnStretch(1, 1)
        self.layout_tab_imagen.setRowStretch(0, 1)
        self.layout_tab_imagen.setRowStretch(1, 1)

        # Pestaña "Búsqueda A*"
        self.tab_a_estrella = QWidget()
        self.tabs.addTab(self.tab_a_estrella, "Búsqueda A*")
        self.layout_tab_a_estrella = QVBoxLayout(self.tab_a_estrella)

        # Pestaña "STRIPS"
        self.tab_strips = QWidget()
        self.tabs.addTab(self.tab_strips, "STRIPS")
        self.layout_tab_strips = QVBoxLayout(self.tab_strips)


        # Widgets de la pestaña "Detección"
        self.label_imagen = QLabel("Imagen")
        self.label_imagen.setAlignment(Qt.AlignCenter)
        self.label_imagen.setFixedSize(300, 300)
        self.label_imagen.setStyleSheet("background-color: lightgray;")

        self.label_contorno = QLabel("Contorno")
        self.label_contorno.setAlignment(Qt.AlignCenter)
        self.label_contorno.setFixedSize(300, 300)
        self.label_contorno.setStyleSheet("background-color: lightgray;")

        self.grafico_knn = Grafico3DCanvas(self.tab_imagen, width=5, height=4)
        self.grafico_kmeans = Grafico3DCanvas(self.tab_imagen, width=5, height=4)

        # Organización de widgets en la pestaña "Detección"
        self.layout_tab_imagen.addWidget(self.label_imagen, 0, 0, alignment=Qt.AlignCenter)
        self.layout_tab_imagen.addWidget(self.label_contorno, 0, 1, alignment=Qt.AlignCenter)
        self.layout_tab_imagen.addWidget(self.grafico_knn, 1, 0, alignment=Qt.AlignCenter)
        self.layout_tab_imagen.addWidget(self.grafico_kmeans, 1, 1, alignment=Qt.AlignCenter)

        # Widgets inferiores (botones, registro, etc.)
        self.label_prediccion = QLabel("Predicción: [Aquí saldrá la clase]")
        self.text_registro = QTextEdit()
        self.text_registro.setReadOnly(True)
        self.text_registro.setStyleSheet("background-color: #F0F0F0;")

        # Botones: Cargar Imagen, Procesar y Cargar Dataset
        self.boton_cargar = QPushButton("Cargar Imagen")
        self.boton_procesar = QPushButton("Procesar")
        self.boton_dataset = QPushButton("Cargar Dataset")

        # Layout inferior
        layout_inferior = QHBoxLayout()
        layout_inferior.addWidget(self.label_prediccion)
        layout_inferior.addWidget(self.boton_cargar)
        layout_inferior.addWidget(self.boton_procesar)
        layout_inferior.addWidget(self.boton_dataset)

        # Agregar layouts al layout principal
        layout_principal.addLayout(layout_inferior)
        layout_principal.addWidget(self.text_registro)

    def mostrar_imagen(self, imagen_cv, label_widget):
        """
        Muestra una imagen OpenCV en un QLabel.
        """
        imagen_rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = imagen_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(imagen_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio)
        label_widget.setPixmap(pixmap)

    def agregar_a_registro(self, texto):
        """
        Agrega un mensaje al registro de texto.
        """
        self.text_registro.append(texto)

    def set_prediccion(self, texto):
        """
        Actualiza la etiqueta de predicción.
        """
        self.label_prediccion.setText(f"Predicción: {texto}")