from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QIcon  # <--- Agrega QIcon aquí
import os # Importar os para manejar rutas seguramente
import cv2
from view.image_view import Grafico3DCanvas
from view.grid_astar import GridAstarWidget
from view.strips_view import StripsView
from view.zoomable_image_label import ZoomableImageLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("P.A.W.S. - Dashboard de Clasificación")
        
        # --- AGREGAR ICONO ---
        # Asumiendo que guardaste 'app_icon.png' en la carpeta 'resource'
        # Usamos os.path para asegurar que encuentre el archivo
        ruta_icono = os.path.join("images/imgApp/", "app_icon.png")
        self.setWindowIcon(QIcon(ruta_icono))

        self.init_ui()

    def init_ui(self):
        widget_central = QWidget()
        layout_principal = QVBoxLayout(widget_central)
        self.setCentralWidget(widget_central)

        self.tabs = QTabWidget()
        layout_principal.addWidget(self.tabs)

        # ------------- Pestaña "Detección" -------------
        self.tab_deteccion = QWidget()
        self.tabs.addTab(self.tab_deteccion, "Detección")
        self.layout_tab_deteccion = QGridLayout(self.tab_deteccion)
        
        # Configuración de estiramiento
        self.layout_tab_deteccion.setColumnStretch(0, 1)
        self.layout_tab_deteccion.setColumnStretch(1, 1)
        self.layout_tab_deteccion.setRowStretch(0, 1)
        self.layout_tab_deteccion.setRowStretch(1, 1)
        
        # --- Fila 0: Imágenes ---
        vbox_foto = QVBoxLayout()
        label_foto_titulo = QLabel("Imagen Original")
        label_foto_titulo.setAlignment(Qt.AlignCenter)
        vbox_foto.addWidget(label_foto_titulo)
        self.label_imagen = ZoomableImageLabel()
        self.label_imagen.setFixedSize(300, 300)
        self.label_imagen.setStyleSheet("background-color: lightgray;")
        vbox_foto.addWidget(self.label_imagen, alignment=Qt.AlignCenter)
        
        vbox_contorno = QVBoxLayout()
        label_contorno_titulo = QLabel("Imagen con Contorno")
        label_contorno_titulo.setAlignment(Qt.AlignCenter)
        vbox_contorno.addWidget(label_contorno_titulo)
        self.label_contorno = ZoomableImageLabel()
        self.label_contorno.setFixedSize(300, 300)
        self.label_contorno.setStyleSheet("background-color: lightgray;")
        vbox_contorno.addWidget(self.label_contorno, alignment=Qt.AlignCenter)

        self.layout_tab_deteccion.addLayout(vbox_foto, 0, 0, alignment=Qt.AlignCenter)
        self.layout_tab_deteccion.addLayout(vbox_contorno, 0, 1, alignment=Qt.AlignCenter)
        
        # --- Fila 1: Gráficos 3D ---
        vbox_knn = QVBoxLayout()
        label_knn = QLabel("Gráfico KNN (3D)")
        label_knn.setAlignment(Qt.AlignCenter)
        vbox_knn.addWidget(label_knn)
        self.grafico_knn = Grafico3DCanvas(self.tab_deteccion, width=5, height=4)
        vbox_knn.addWidget(self.grafico_knn)
        
        vbox_kmeans = QVBoxLayout()
        label_kmeans = QLabel("Gráfico K-means (3D)")
        label_kmeans.setAlignment(Qt.AlignCenter)
        vbox_kmeans.addWidget(label_kmeans)
        self.grafico_kmeans = Grafico3DCanvas(self.tab_deteccion, width=5, height=4)
        vbox_kmeans.addWidget(self.grafico_kmeans)
        
        self.layout_tab_deteccion.addLayout(vbox_knn, 1, 0, alignment=Qt.AlignCenter)
        self.layout_tab_deteccion.addLayout(vbox_kmeans, 1, 1, alignment=Qt.AlignCenter)

        # --- Fila 2 (NUEVA): Botones para Gráficos 2D ---
        hbox_botones_graficos = QHBoxLayout()
        self.btn_comp1 = QPushButton("Ver Solidez vs Hu0")
        self.btn_comp2 = QPushButton("Ver Circularidad vs Hu0")
        self.btn_comp3 = QPushButton("Ver Solidez vs Circularidad")
        
        hbox_botones_graficos.addWidget(self.btn_comp1)
        hbox_botones_graficos.addWidget(self.btn_comp2)
        hbox_botones_graficos.addWidget(self.btn_comp3)
        
        # Añadir esta fila al grid de detección (ocupando ambas columnas)
        self.layout_tab_deteccion.addLayout(hbox_botones_graficos, 2, 0, 1, 2, alignment=Qt.AlignCenter)


        # ------------- Pestaña "Búsqueda A*" -------------
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
        
        # Layout para botones de A*
        astar_buttons_layout = QHBoxLayout()
        self.button_astar = QPushButton("Calcular Ruta A*")
        self.button_astar_clear = QPushButton("Limpiar Grilla")
        astar_buttons_layout.addWidget(self.button_astar)
        astar_buttons_layout.addWidget(self.button_astar_clear)
        self.layout_tab_astar.addLayout(astar_buttons_layout)

        # ------------- Pestaña "STRIPS" -------------
        self.tab_strips = QWidget()
        self.tabs.addTab(self.tab_strips, "STRIPS")
        self.layout_tab_strips = QHBoxLayout(self.tab_strips)
        self.strips_view = StripsView()
        self.layout_tab_strips.addWidget(self.strips_view)
        vbox_strips = QVBoxLayout()
        
        label_domain = QLabel("Dominio STRIPS")
        vbox_strips.addWidget(label_domain)
        self.text_strips_domain = QTextEdit()
        self.text_strips_domain.setReadOnly(True)
        vbox_strips.addWidget(self.text_strips_domain)
        
        label_problem = QLabel("Problema STRIPS")
        vbox_strips.addWidget(label_problem)
        self.text_strips_problem = QTextEdit()
        self.text_strips_problem.setReadOnly(True)
        vbox_strips.addWidget(self.text_strips_problem)
        
        self.layout_tab_strips.addLayout(vbox_strips)

        # ------------- Área inferior -------------
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
        self.text_registro = QTextEdit()
        self.text_registro.setReadOnly(True)
        layout_principal.addWidget(self.text_registro)

    def mostrar_imagen(self, imagen_cv, label_widget):
        imagen_rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = imagen_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(imagen_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio)
        label_widget.setPixmap(pixmap)

    def agregar_a_registro(self, texto):
        self.text_registro.append(texto)

    def set_prediccion(self, texto):
        self.label_prediccion.setText(f"Predicción: {texto}")