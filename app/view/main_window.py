from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
from view.image_view import Grafico3DCanvas
from view.grid_astar import GridAstarWidget
from view.strips_view import StripsView
from view.zoomable_image_label import ZoomableImageLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard VisionAR")
        self.init_ui()

    def init_ui(self):
        # Widget central y layout principal
        widget_central = QWidget()
        layout_principal = QVBoxLayout(widget_central)
        self.setCentralWidget(widget_central)

        # TabWidget con las pestañas
        self.tabs = QTabWidget()
        layout_principal.addWidget(self.tabs)

        # --- Pestaña "Detección" ---
        self.tab_deteccion = QWidget()
        self.tabs.addTab(self.tab_deteccion, "Detección")
        self.layout_tab_deteccion = QGridLayout(self.tab_deteccion)
        self.layout_tab_deteccion.setColumnStretch(0, 1)
        self.layout_tab_deteccion.setColumnStretch(1, 1)
        self.layout_tab_deteccion.setRowStretch(0, 1)
        self.layout_tab_deteccion.setRowStretch(1, 1)
        self.label_imagen = ZoomableImageLabel()
        self.label_imagen.setFixedSize(300, 300)
        self.label_imagen.setStyleSheet("background-color: lightgray;")
        self.label_contorno = ZoomableImageLabel()
        self.label_contorno.setFixedSize(300, 300)
        self.label_contorno.setStyleSheet("background-color: lightgray;")
        self.layout_tab_deteccion.addWidget(self.label_imagen, 0, 0, alignment=Qt.AlignCenter)
        self.layout_tab_deteccion.addWidget(self.label_contorno, 0, 1, alignment=Qt.AlignCenter)
        # Gráficos 3D con títulos (cada uno en un QVBoxLayout)
        vbox_knn = QVBoxLayout()
        label_knn = QLabel("Gráfico KNN")
        label_knn.setAlignment(Qt.AlignCenter)
        vbox_knn.addWidget(label_knn)
        self.grafico_knn = Grafico3DCanvas(self.tab_deteccion, width=5, height=4)
        vbox_knn.addWidget(self.grafico_knn)
        vbox_kmeans = QVBoxLayout()
        label_kmeans = QLabel("Gráfico K-means")
        label_kmeans.setAlignment(Qt.AlignCenter)
        vbox_kmeans.addWidget(label_kmeans)
        self.grafico_kmeans = Grafico3DCanvas(self.tab_deteccion, width=5, height=4)
        vbox_kmeans.addWidget(self.grafico_kmeans)
        self.layout_tab_deteccion.addLayout(vbox_knn, 1, 0, alignment=Qt.AlignCenter)
        self.layout_tab_deteccion.addLayout(vbox_kmeans, 1, 1, alignment=Qt.AlignCenter)

        # --- Pestaña "Búsqueda A*" ---
        self.tab_astar = QWidget()
        self.tabs.addTab(self.tab_astar, "Búsqueda A*")
        self.layout_tab_astar = QVBoxLayout(self.tab_astar)
        default_grid = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        self.grid_widget = GridAstarWidget(default_grid)
        self.layout_tab_astar.addWidget(self.grid_widget, alignment=Qt.AlignCenter)
        self.button_astar = QPushButton("Calcular Ruta A*")
        self.layout_tab_astar.addWidget(self.button_astar, alignment=Qt.AlignCenter)

        # --- Pestaña "STRIPS" ---
        self.tab_strips = QWidget()
        self.tabs.addTab(self.tab_strips, "STRIPS")
        self.layout_tab_strips = QHBoxLayout(self.tab_strips)
        # Izquierda: widget STRIPS para definir y visualizar el orden de cajas
        self.strips_view = StripsView()
        self.layout_tab_strips.addWidget(self.strips_view)
        # Derecha: dos áreas de texto en un layout vertical para el dominio y el problema STRIPS
        vbox_strips = QVBoxLayout()
        self.text_strips_domain = QTextEdit()
        self.text_strips_domain.setReadOnly(True)
        self.text_strips_domain.setPlaceholderText("Dominio STRIPS")
        vbox_strips.addWidget(self.text_strips_domain)
        self.text_strips_problem = QTextEdit()
        self.text_strips_problem.setReadOnly(True)
        self.text_strips_problem.setPlaceholderText("Problema STRIPS")
        vbox_strips.addWidget(self.text_strips_problem)
        self.layout_tab_strips.addLayout(vbox_strips)

        # --- Área inferior común (etiqueta de predicción y botones) ---
        self.label_prediccion = QLabel("Predicción: [Aquí saldrá la clase]")
        self.boton_cargar = QPushButton("Cargar Imagen")
        self.boton_procesar = QPushButton("Procesar")
        self.boton_dataset = QPushButton("Cargar Dataset")
        layout_inferior = QHBoxLayout()
        layout_inferior.addWidget(self.label_prediccion)
        layout_inferior.addWidget(self.boton_cargar)
        layout_inferior.addWidget(self.boton_procesar)
        layout_inferior.addWidget(self.boton_dataset)
        layout_principal.addLayout(layout_inferior)

        # --- Área de texto para registro de mensajes ---
        self.text_registro = QTextEdit()
        self.text_registro.setReadOnly(True)
        self.text_registro.setStyleSheet("background-color: #F0F0F0;")
        layout_principal.addWidget(self.text_registro)

    def mostrar_imagen(self, imagen_cv, label_widget):
        """Muestra una imagen OpenCV en un QLabel."""
        imagen_rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = imagen_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(imagen_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio)
        label_widget.setPixmap(pixmap)

    def agregar_a_registro(self, texto):
        """Agrega un mensaje al área de registro."""
        self.text_registro.append(texto)

    def set_prediccion(self, texto):
        """Actualiza la etiqueta de predicción."""
        self.label_prediccion.setText(f"Predicción: {texto}")
