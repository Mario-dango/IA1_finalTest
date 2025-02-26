from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit, QGridLayout
)
from PyQt5.QtCore import Qt                    # Importa constantes y funciones de Qt (por ejemplo, AlignCenter)
from PyQt5.QtGui import QPixmap, QImage          # Importa clases para manejo de imágenes
import cv2                                     # Importa OpenCV para procesamiento de imágenes
from view.image_view import Grafico3DCanvas     # Importa el canvas para gráficos 3D (KNN, K-means)
from view.grid_astar import GridAstarWidget       # Importa el widget interactivo para A*
from view.strips_view import StripsView           # Importa el widget interactivo para STRIPS
from view.zoomable_image_label import ZoomableImageLabel  # Importa el QLabel con zoom para imágenes

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()                        # Llama al constructor de QMainWindow
        self.setWindowTitle("Dashboard VisionAR") # Establece el título de la ventana
        self.init_ui()                             # Inicializa la interfaz gráfica

    def init_ui(self):
        # Crea el widget central y define un layout vertical principal
        widget_central = QWidget()
        layout_principal = QVBoxLayout(widget_central)
        self.setCentralWidget(widget_central)       # Asigna el widget central a la ventana

        # Crea un QTabWidget para organizar las pestañas de la aplicación
        self.tabs = QTabWidget()
        layout_principal.addWidget(self.tabs)

        # ------------- Pestaña "Detección" -------------
        self.tab_deteccion = QWidget()               # Crea el widget para la pestaña "Detección"
        self.tabs.addTab(self.tab_deteccion, "Detección")  # Añade la pestaña al QTabWidget
        self.layout_tab_deteccion = QGridLayout(self.tab_deteccion)  # Define un layout en forma de cuadrícula
        self.layout_tab_deteccion.setColumnStretch(0, 1)  # Configura la distribución horizontal
        self.layout_tab_deteccion.setColumnStretch(1, 1)
        self.layout_tab_deteccion.setRowStretch(0, 1)     # Configura la distribución vertical
        self.layout_tab_deteccion.setRowStretch(1, 1)
        
        # Se crea un QVBoxLayout para la imagen original, que incluirá un título y el widget de imagen
        vbox_foto = QVBoxLayout()
        label_foto_titulo = QLabel("Imagen Original")
        label_foto_titulo.setAlignment(Qt.AlignCenter)
        vbox_foto.addWidget(label_foto_titulo)
        # Se crea un ZoomableImageLabel para mostrar la imagen original con funcionalidad de zoom
        self.label_imagen = ZoomableImageLabel()
        self.label_imagen.setFixedSize(300, 300)      # Fija el tamaño del widget
        self.label_imagen.setStyleSheet("background-color: lightgray;")  # Define un fondo gris claro
        vbox_foto.addWidget(self.label_imagen, alignment=Qt.AlignCenter)
        
        # Se crea otro QVBoxLayout para la imagen con contorno, con su título correspondiente
        vbox_contorno = QVBoxLayout()
        label_contorno_titulo = QLabel("Imagen con Contorno")
        label_contorno_titulo.setAlignment(Qt.AlignCenter)
        vbox_contorno.addWidget(label_contorno_titulo)
        # Se crea otro ZoomableImageLabel para mostrar la imagen con contorno dibujado
        self.label_contorno = ZoomableImageLabel()
        self.label_contorno.setFixedSize(300, 300)
        self.label_contorno.setStyleSheet("background-color: lightgray;")
        vbox_contorno.addWidget(self.label_contorno, alignment=Qt.AlignCenter)

        # Se añaden los layouts verticales al layout en forma de grid (fila 0)
        self.layout_tab_deteccion.addLayout(vbox_foto, 0, 0, alignment=Qt.AlignCenter)
        self.layout_tab_deteccion.addLayout(vbox_contorno, 0, 1, alignment=Qt.AlignCenter)
        
        # Se crean dos canvas para gráficos 3D: uno para KNN y otro para K-means
        # Cada canvas se coloca en un layout vertical junto con un QLabel que actúa de título
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
        
        # Se añade cada layout al grid en la segunda fila, centrados
        self.layout_tab_deteccion.addLayout(vbox_knn, 1, 0, alignment=Qt.AlignCenter)
        self.layout_tab_deteccion.addLayout(vbox_kmeans, 1, 1, alignment=Qt.AlignCenter)

        # ------------- Pestaña "Búsqueda A*" -------------
        self.tab_astar = QWidget()                    # Crea el widget para la pestaña "Búsqueda A*"
        self.tabs.addTab(self.tab_astar, "Búsqueda A*") # Añade la pestaña
        self.layout_tab_astar = QVBoxLayout(self.tab_astar)  # Define un layout vertical para esta pestaña
        # Define un grid de ejemplo para representar el entorno de búsqueda
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
        # Crea el widget de la grilla para A* con el grid definido
        self.grid_widget = GridAstarWidget(default_grid)
        self.layout_tab_astar.addWidget(self.grid_widget, alignment=Qt.AlignCenter)
        # Añade un botón para calcular la ruta con A*
        self.button_astar = QPushButton("Calcular Ruta A*")
        self.layout_tab_astar.addWidget(self.button_astar, alignment=Qt.AlignCenter)

        # ------------- Pestaña "STRIPS" -------------
        self.tab_strips = QWidget()                    # Crea el widget para la pestaña "STRIPS"
        self.tabs.addTab(self.tab_strips, "STRIPS")     # Añade la pestaña
        # Define un layout horizontal para organizar el widget STRIPS y las áreas de texto al lado
        self.layout_tab_strips = QHBoxLayout(self.tab_strips)
        # A la izquierda, se coloca el widget STRIPS para definir el orden de las cajas
        self.strips_view = StripsView()
        self.layout_tab_strips.addWidget(self.strips_view)
        # A la derecha, se crea un layout vertical para dos QTextEdit (uno para el dominio y otro para el problema)
        vbox_strips = QVBoxLayout()
        self.text_strips_domain = QTextEdit()
        self.text_strips_domain.setReadOnly(True)    # Se establece en modo solo lectura
        self.text_strips_domain.setPlaceholderText("Dominio STRIPS")  # Texto de sugerencia
        vbox_strips.addWidget(self.text_strips_domain)
        self.text_strips_problem = QTextEdit()
        self.text_strips_problem.setReadOnly(True)
        self.text_strips_problem.setPlaceholderText("Problema STRIPS")
        vbox_strips.addWidget(self.text_strips_problem)
        self.layout_tab_strips.addLayout(vbox_strips)   # Se añade el layout vertical al layout horizontal de la pestaña STRIPS

        # ------------- Área inferior común -------------
        # Se crea una etiqueta para mostrar la predicción
        self.label_prediccion = QLabel("Predicción: [Aquí saldrá la clase]")
        # Se crean botones para cargar imagen, procesar imagen y cargar dataset
        self.boton_cargar = QPushButton("Cargar Imagen")
        self.boton_procesar = QPushButton("Procesar")
        self.boton_dataset = QPushButton("Cargar Dataset")
        # Se organiza todo en un layout horizontal
        layout_inferior = QHBoxLayout()
        layout_inferior.addWidget(self.label_prediccion)
        layout_inferior.addWidget(self.boton_cargar)
        layout_inferior.addWidget(self.boton_procesar)
        layout_inferior.addWidget(self.boton_dataset)
        layout_principal.addLayout(layout_inferior)   # Se añade el layout inferior al layout principal
        # Se añade un QTextEdit para registro de mensajes (para debug y seguimiento)
        self.text_registro = QTextEdit()
        self.text_registro.setReadOnly(True)
        self.text_registro.setStyleSheet("background-color: #F0F0F0;")
        layout_principal.addWidget(self.text_registro)

    def mostrar_imagen(self, imagen_cv, label_widget):
        """
        Convierte una imagen OpenCV (BGR) a QImage y la muestra en el QLabel especificado.
        :param imagen_cv: Imagen en formato BGR.
        :param label_widget: QLabel en el que se mostrará la imagen.
        """
        imagen_rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)  # Convierte la imagen a formato RGB
        h, w, ch = imagen_rgb.shape                     # Obtiene la altura, ancho y número de canales
        bytes_per_line = ch * w                           # Calcula los bytes por línea
        qimg = QImage(imagen_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)  # Crea un QImage
        pixmap = QPixmap.fromImage(qimg)                  # Convierte el QImage a QPixmap
        pixmap = pixmap.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio)  # Escala el pixmap manteniendo la relación de aspecto
        label_widget.setPixmap(pixmap)                    # Muestra el pixmap en el QLabel

    def agregar_a_registro(self, texto):
        """
        Agrega un mensaje al área de registro (QTextEdit) para mostrar información de debug o estado.
        :param texto: Cadena de texto a agregar.
        """
        self.text_registro.append(texto)

    def set_prediccion(self, texto):
        """
        Actualiza la etiqueta de predicción con el texto dado.
        :param texto: Cadena de texto que representa la predicción.
        """
        self.label_prediccion.setText(f"Predicción: {texto}")
