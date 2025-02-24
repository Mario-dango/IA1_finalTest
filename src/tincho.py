import sys
import os
import numpy as np
import cv2
import math

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFileDialog, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

# Para la parte de matplotlib en PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # Necesario para 3D
# if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
#     del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
# # Forzar el uso del plugin xcb
# os.environ["QT_QPA_PLATFORM"] = "xcb"

###############################################################################
#                                MODELO
###############################################################################
class Modelo(object):
    """
    El Modelo se encarga de almacenar los datos (características extraídas),
    de implementar los algoritmos manuales para KNN y K-means, y de normalizar.
    """
    def __init__(self):
        # Estructura: [circularidad, aspect_ratio, hu0, etiqueta]
        self.datos_entrenamiento = []
        self.centroides = None  # Centroides en el espacio normalizado
        self.mean = None
        self.std = None

    def calcular_caracteristicas(self, imagen):
        """
        Dada una imagen (BGR), retorna:
         - Circularidad
         - Aspect Ratio
         - Excentricidad (calculada pero no usada en la clasificación 3D)
         - Primer momento de Hu (hu0)
        Se utiliza un preprocesamiento para evitar que la sombra se detecte como borde.
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # Suavizado para reducir ruido y sombras
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        # Threshold adaptativo
        # Parámetros ajustados
        blockSize = 25  # Tamaño de la ventana (debe ser impar)
        C = 5           # Constante para ajustar el umbral

        # Umbralización adaptativa
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C
        )

        # Operación morfológica de cierre para eliminar pequeños huecos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # Tamaño del kernel
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Buscar contornos
        contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        # gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # t_lower = 100  # Lower Threshold, bajo de este nivel no detecta el contorno.
        # t_upper = 200  # Upper threshold, entre lower y upper se dibuja solo si el contorno conecta con uno de valor mayor a upper
        # aperture_size = 3  # Aperture size 
        # L2Gradient = False # Boolean 
        # img = cv2.GaussianBlur(gray, (5,5), 0)
        # sizeKernel = np.ones((4,3), np.uint8); iterations=2
        # morpho=cv2.dilate(img, sizeKernel, iterations)
        # imga = cv2.Canny(morpho, t_lower, t_upper, apertureSize = aperture_size,  L2gradient = L2Gradient)
        # Encuentra los contornos en la imagen filtrada por Canny
        # contornos,_ = cv2.findContours(imga, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            return 0, 0, 0, 0
        
        contorno = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        circularidad = 4 * math.pi * (area / (perimetro * perimetro)) if perimetro != 0 else 0

        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = w / float(h) if h != 0 else 0

        # Calcular excentricidad mediante fitEllipse
        excentricidad = 0
        if len(contorno) >= 5:
            (x_elipse, y_elipse), (eje1, eje2), angulo = cv2.fitEllipse(contorno)
            major = max(eje1, eje2)
            minor = min(eje1, eje2)
            if major != 0:
                ratio = (minor / major)**2
                if ratio > 1:
                    ratio = 1
                excentricidad = math.sqrt(1 - ratio)
            else:
                excentricidad = 0

        # Primer momento de Hu
        momentos = cv2.moments(contorno)
        hu = cv2.HuMoments(momentos)
        hu0 = -1 if hu is None else hu[0][0]
        
        return circularidad, aspect_ratio, excentricidad, hu0

    def actualizar_normalizacion(self):
        """
        Calcula media y std de las tres características (circularidad, aspect_ratio, hu0)
        en el dataset de entrenamiento.
        """
        if not self.datos_entrenamiento:
            return
        datos = np.array([fila[:3] for fila in self.datos_entrenamiento], dtype=np.float32)
        self.mean = np.mean(datos, axis=0)
        self.std = np.std(datos, axis=0)
        self.std[self.std == 0] = 1  # Evitar división por 0

    def normalizar(self, features):
        """
        Normaliza un vector de características usando la media y std calculadas.
        """
        if self.mean is None or self.std is None:
            return features
        return (np.array(features) - self.mean) / self.std

    # <-- Cambio: Agregamos la función para "desnormalizar" una característica
    def desnormalizar(self, features):
        """
        Des-normaliza un vector de características (inverso de normalizar).
        """
        if self.mean is None or self.std is None:
            return features
        return features * self.std + self.mean

    def knn_manual(self, nuevo_punto, k=3):
        """
        Implementación manual de KNN usando las 3 características (normalizadas).
        """
        nuevo_norm = self.normalizar(nuevo_punto)
        distancias = []
        for fila in self.datos_entrenamiento:
            features = fila[:3]
            features_norm = self.normalizar(features)
            dist = np.linalg.norm(nuevo_norm - features_norm)
            distancias.append((dist, fila[3]))
        distancias.sort(key=lambda x: x[0])
        vecinos = distancias[:k]
        conteo = {}
        for d, etiq in vecinos:
            conteo[etiq] = conteo.get(etiq, 0) + 1
        etiqueta_ganadora = max(conteo, key=conteo.get)
        return etiqueta_ganadora

    def kmeans_manual(self, datos, k=4, max_iter=100):
        """
        Implementación manual de K-means sobre datos: lista de [circ, aspect_ratio, hu0].
        Se normalizan los datos.
        Retorna:
          - centroides (en el espacio NORMALIZADO)
          - asignaciones (para cada punto)
        """
        datos_np = np.array([self.normalizar(d) for d in datos], dtype=np.float32)
        N = datos_np.shape[0]
        if N < k:
            k = N
        idx_random = np.random.choice(N, k, replace=False)
        centroides = datos_np[idx_random, :]
        
        for _ in range(max_iter):
            asignaciones = []
            for punto in datos_np:
                distancias = [np.linalg.norm(punto - c) for c in centroides]
                cluster_id = np.argmin(distancias)
                asignaciones.append(cluster_id)
            asignaciones = np.array(asignaciones)
            nuevos_centroides = []
            for cluster_id in range(k):
                cluster_puntos = datos_np[asignaciones == cluster_id]
                if len(cluster_puntos) > 0:
                    nuevo_c = np.mean(cluster_puntos, axis=0)
                else:
                    nuevo_c = datos_np[np.random.choice(N)]
                nuevos_centroides.append(nuevo_c)
            nuevos_centroides = np.array(nuevos_centroides)
            if np.allclose(centroides, nuevos_centroides, atol=1e-6):
                break
            centroides = nuevos_centroides
        
        self.centroides = centroides  # Guardamos los centroides en el espacio normalizado
        return centroides, asignaciones

    # <-- Cambio: Función para obtener los centroides "desnormalizados"
    def get_centroids_unnorm(self):
        if self.centroides is None:
            return None
        centroids_unnorm = []
        for c in self.centroides:
            c_desnorm = self.desnormalizar(c)
            centroids_unnorm.append(c_desnorm)
        return np.array(centroids_unnorm)

    def cargar_dataset(self, ruta_dataset):
        """
        Carga un dataset desde una carpeta. Se espera que dentro de 'ruta_dataset'
        existan subcarpetas con el nombre de cada clase, y que cada subcarpeta contenga imágenes.
        """
        self.datos_entrenamiento = []  # Limpiar dataset anterior
        extensiones = ('.png', '.jpg', '.jpeg', '.bmp')
        for subfolder in os.listdir(ruta_dataset):
            ruta_sub = os.path.join(ruta_dataset, subfolder)
            if os.path.isdir(ruta_sub):
                # subfolder es el nombre de la clase (por ej. "Tuerca", "Tornillo", etc.)
                for archivo in os.listdir(ruta_sub):
                    if archivo.lower().endswith(extensiones):
                        ruta_img = os.path.join(ruta_sub, archivo)
                        imagen = cv2.imread(ruta_img)
                        if imagen is not None:
                            circ, asp, exc, hu0 = self.calcular_caracteristicas(imagen)
                            if circ != 0 or asp != 0:
                                self.datos_entrenamiento.append([circ, asp, hu0, subfolder])
        self.actualizar_normalizacion()
        return len(self.datos_entrenamiento)

###############################################################################
#                                VISTA
###############################################################################
class Grafico3DCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)

    def plot_puntos(self, datos, centroides=None, asignaciones=None, titulo="Gráfico 3D"):
        """
        datos: lista de [circ, aspect, hu0, etiqueta] (en el espacio ORIGINAL)
        centroides: np.array con shape (k,3) (en el espacio ORIGINAL, si quieres que se vean en la escala real)
        """
        self.axes.clear()
        self.axes.set_title(titulo)
        self.axes.set_xlabel("Circularidad")
        self.axes.set_ylabel("Aspect Ratio")
        self.axes.set_zlabel("Hu[0]")

        if asignaciones is not None:
            # asume que "datos" y "centroides" están en la misma escala (original).
            asignaciones = np.array(asignaciones)
            for cluster_id in np.unique(asignaciones):
                cluster_puntos = np.array(
                    [d[:3] for i, d in enumerate(datos) if asignaciones[i] == cluster_id]
                )
                if len(cluster_puntos) > 0:
                    self.axes.scatter(cluster_puntos[:, 0],
                                      cluster_puntos[:, 1],
                                      cluster_puntos[:, 2],
                                      label=f"Cluster {cluster_id}")
            if centroides is not None:
                self.axes.scatter(centroides[:, 0],
                                  centroides[:, 1],
                                  centroides[:, 2],
                                  c='black', marker='X', s=100, label="Centroides")
        else:
            # Graficamos según la etiqueta
            etiquetas = set([d[3] for d in datos])
            colores = ["red", "green", "blue", "orange", "magenta", "cyan"]
            col_idx = 0
            for etiq in etiquetas:
                subset = np.array([d[:3] for d in datos if d[3] == etiq])
                self.axes.scatter(subset[:, 0], subset[:, 1], subset[:, 2],
                                  color=colores[col_idx % len(colores)],
                                  label=str(etiq))
                col_idx += 1
        
        self.axes.legend()
        self.draw()

class Vista(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard VisionAR")
        widget_central = QWidget()
        layout_principal = QVBoxLayout(widget_central)

        self.tabs = QTabWidget()
        layout_principal.addWidget(self.tabs)

        # Pestaña "Imagen/Contorno"
        self.tab_imagen = QWidget()
        self.tabs.addTab(self.tab_imagen, "Detección")
        self.layout_tab_imagen = QGridLayout(self.tab_imagen)
        # Asigna más espacio a las columnas (0 y 1)
        self.layout_tab_imagen.setColumnStretch(0, 1)
        self.layout_tab_imagen.setColumnStretch(1, 1)

        # Si deseas que las filas también se expandan, puedes hacerlo:
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

        # Widgets de la pestaña "Imagen/Contorno"
        self.label_imagen = QLabel("Imagen")
        self.label_imagen.setAlignment(Qt.AlignCenter)
        self.label_imagen.setFixedSize(300, 300)
        self.label_imagen.setStyleSheet("background-color: lightgray;")
        self.label_contorno = QLabel("Contorno")
        self.label_contorno.setAlignment(Qt.AlignCenter)
        self.label_contorno.setFixedSize(300, 300)
        self.label_contorno.setStyleSheet("background-color: lightgray;")

        # Gráficos de KNN y K-Means (ahora en la pestaña "Imagen/Contorno")
        self.grafico_knn = Grafico3DCanvas(self.tab_imagen, width=5, height=4)
        self.grafico_kmeans = Grafico3DCanvas(self.tab_imagen, width=5, height=4)

        # Organización de widgets en la pestaña "Imagen/Contorno"
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

        layout_inferior = QHBoxLayout()
        layout_inferior.addWidget(self.label_prediccion)
        layout_inferior.addWidget(self.boton_cargar)
        layout_inferior.addWidget(self.boton_procesar)
        layout_inferior.addWidget(self.boton_dataset)

        layout_principal.addLayout(layout_inferior)
        layout_principal.addWidget(self.text_registro)

        self.setCentralWidget(widget_central)

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

###############################################################################
#                                CONTROLADOR
###############################################################################
class Controlador:
    def __init__(self, modelo, vista):
        self.modelo = modelo
        self.vista = vista

        self.vista.boton_cargar.clicked.connect(self.cargar_imagen)
        self.vista.boton_procesar.clicked.connect(self.procesar_imagen)
        self.vista.boton_dataset.clicked.connect(self.cargar_dataset)

        self.modelo.datos_entrenamiento = []
        self.imagen_actual = None
        self.vista.agregar_a_registro("Dataset vacío. Carga un dataset para mejorar la clasificación.")

    def cargar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(None, "Seleccionar imagen", "", "Images (*.png *.jpg *.bmp)")
        if ruta:
            self.imagen_actual = cv2.imread(ruta)
            if self.imagen_actual is not None:
                self.vista.mostrar_imagen(self.imagen_actual, self.vista.label_imagen)
                self.vista.agregar_a_registro(f"Imagen cargada: {ruta}")
            else:
                self.vista.agregar_a_registro("Error al cargar la imagen.")
        else:
            self.vista.agregar_a_registro("No se seleccionó ninguna imagen.")

    def cargar_dataset(self):
        ruta = QFileDialog.getExistingDirectory(None, "Seleccionar carpeta del dataset")
        if ruta:
            cantidad = self.modelo.cargar_dataset(ruta)
            self.vista.agregar_a_registro(f"Dataset cargado desde: {ruta} con {cantidad} muestras.")
            if self.modelo.datos_entrenamiento:
                # Graficar KNN (sin asignaciones)
                self.vista.grafico_knn.plot_puntos(
                    self.modelo.datos_entrenamiento,  # en escala original
                    titulo="Dataset (KNN)"
                )
                # Hacer K-means
                datos_3d = [fila[:3] for fila in self.modelo.datos_entrenamiento]
                _, asignaciones = self.modelo.kmeans_manual(datos_3d, k=4, max_iter=10)
                # Des-normalizar los centroides para graficar
                centroides_unnorm = self.modelo.get_centroids_unnorm()
                self.vista.grafico_kmeans.plot_puntos(
                    self.modelo.datos_entrenamiento,  # en escala original
                    centroides=centroides_unnorm,      # en escala original
                    asignaciones=asignaciones,
                    titulo="Dataset (K-means)"
                )
        else:
            self.vista.agregar_a_registro("No se seleccionó carpeta para el dataset.")

    def procesar_imagen(self):
        if self.imagen_actual is None:
            self.vista.agregar_a_registro("No hay imagen para procesar. Carga una primero.")
            return

        circ, asp, exc, hu0 = self.modelo.calcular_caracteristicas(self.imagen_actual)
        self.vista.agregar_a_registro(
            f"Características extraídas -> Circularidad: {circ:.3f}, "
            f"Aspect Ratio: {asp:.3f}, Excentricidad: {exc:.3f}, Hu[0]: {hu0:.6f}"
        )

        contorno_img = self.generar_imagen_contorno(self.imagen_actual)
        self.vista.mostrar_imagen(contorno_img, self.vista.label_contorno)

        # Clasificación KNN con normalización
        etiqueta_knn = self.modelo.knn_manual([circ, asp, hu0])
        self.vista.agregar_a_registro(f"KNN dice que es: {etiqueta_knn}")

        # Agregar nuevo punto al dataset (temporal) para K-means
        datos_con_nuevo = self.modelo.datos_entrenamiento.copy()
        datos_con_nuevo.append([circ, asp, hu0, "Desconocido"])
        datos_3d = [fila[:3] for fila in datos_con_nuevo]

        # K-means en el espacio normalizado
        _, asignaciones = self.modelo.kmeans_manual(datos_3d, k=4, max_iter=20)
        cluster_nuevo = asignaciones[-1]
        
        # Mapear cada cluster al nombre mayoritario según el dataset (excluyendo "Desconocido")
        cluster_mapping = {}
        clusters = np.unique(asignaciones[:-1])  # solo de las muestras de entrenamiento
        for cl in clusters:
            etiquetas = [datos_con_nuevo[i][3] for i in range(len(asignaciones)-1) if asignaciones[i]==cl]
            if etiquetas:
                # Votación mayoritaria
                cluster_mapping[cl] = max(set(etiquetas), key=etiquetas.count)
        # Nombre del cluster al que fue asignado
        cluster_nombre = cluster_mapping.get(cluster_nuevo, "Desconocido")

        # Des-normalizar centroides para graficarlos en el espacio original
        centroides_unnorm = self.modelo.get_centroids_unnorm()
        self.vista.grafico_kmeans.plot_puntos(
            datos_con_nuevo,               # en escala original
            centroides=centroides_unnorm,  # en escala original
            asignaciones=asignaciones,
            titulo="K-means (con nuevo punto)"
        )

        mensaje = f"Es {etiqueta_knn} estúpido (K-means sugiere '{cluster_nombre}')."
        self.vista.set_prediccion(mensaje)

    def generar_imagen_contorno(self, imagen_cv):
        gray = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno_img = imagen_cv.copy()

        # gray = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
        # t_lower = 100  # Lower Threshold, bajo de este nivel no detecta el contorno.
        # t_upper = 200  # Upper threshold, entre lower y upper se dibuja solo si el contorno conecta con uno de valor mayor a upper
        # aperture_size = 3  # Aperture size 
        # L2Gradient = False # Boolean 
        # img = cv2.GaussianBlur(gray, (5,5), 0)
        # sizeKernel = np.ones((4,3), np.uint8); iterations=2
        # morpho=cv2.dilate(img, sizeKernel, iterations)
        # imga = cv2.Canny(morpho, t_lower, t_upper, apertureSize = aperture_size,  L2gradient = L2Gradient)
        # # Encuentra los contornos en la imagen filtrada por Canny
        # contornos,_ = cv2.findContours(imga, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(contorno_img, contornos, -1, (0, 255, 0), 2)
        return contorno_img

###############################################################################
#                                MAIN
###############################################################################
def main():
    app = QApplication(sys.argv)
    modelo = Modelo()
    vista = Vista()
    controlador = Controlador(modelo, vista)
    vista.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()