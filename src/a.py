import sys
import os
import cv2
import numpy as np
from math import pi, sqrt, log10
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# -------------------- MODELO --------------------

class FeatureExtractor:
    def extract_features(self, image):
        """
        Extrae las siguientes características del objeto:
        - Circularidad: 4*pi*Area/(Perímetro^2)
        - Aspect Ratio: ancho/alto del rectángulo delimitador
        - Excentricidad: a partir del ajuste de una elipse
        - Hu Moment 1: primer momento de Hu (transformado a escala logarítmica)
        """
        # Convertir a escala de grises y suavizar
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Umbralización (invertido para objetos oscuros sobre fondo claro o viceversa)
        ret, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # Encontrar contornos
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return np.array([0, 0, 0, 0])
        # Seleccionar el contorno más grande
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
        
        # Aspect Ratio a partir del rectángulo delimitador
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Excentricidad a partir de una elipse (si hay suficientes puntos)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            eccentricity = sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis != 0 else 0
        else:
            eccentricity = 0

        # Cálculo de momentos de Hu
        moments = cv2.moments(cnt)
        huMoments = cv2.HuMoments(moments).flatten()
        # Usamos el primer Hu moment y lo transformamos (para mayor estabilidad numérica)
        if huMoments[0] != 0:
            hu1 = -np.sign(huMoments[0]) * log10(abs(huMoments[0]))
        else:
            hu1 = 0

        return np.array([circularity, aspect_ratio, eccentricity, hu1])

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def predict(self, train_features, train_labels, test_feature):
        """
        Implementación manual de KNN:
        - Calcula la distancia euclidiana entre el test_feature y todos los train_features.
        - Selecciona los k vecinos más cercanos.
        - Retorna la etiqueta mayoritaria.
        """
        distances = np.linalg.norm(train_features - test_feature, axis=1)
        k_indices = distances.argsort()[:self.k]
        k_labels = train_labels[k_indices]
        # Conteo de votos
        counts = {}
        for label in k_labels:
            counts[label] = counts.get(label, 0) + 1
        prediction = max(counts, key=counts.get)
        return prediction

class KMeansClustering:
    def __init__(self, k=4, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, features):
        """
        Implementación manual de K-means:
        - Inicializa centroides aleatoriamente a partir de los datos.
        - Itera asignando cada punto al centroide más cercano y actualizando los centroides.
        """
        np.random.seed(42)
        initial_indices = np.random.choice(len(features), self.k, replace=False)
        centroids = features[initial_indices]
        for i in range(self.max_iter):
            # Calcular distancias y asignar clusters
            distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
            cluster_labels = np.argmin(distances, axis=1)
            new_centroids = []
            for j in range(self.k):
                if np.any(cluster_labels == j):
                    new_centroids.append(features[cluster_labels == j].mean(axis=0))
                else:
                    new_centroids.append(centroids[j])
            new_centroids = np.array(new_centroids)
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        return cluster_labels, centroids

class Model:
    def __init__(self):
        self.images = []       # Lista de tuplas: (imagen, etiqueta)
        self.features = None   # Matriz de características
        self.labels = None     # Etiquetas correspondientes
        self.logs = []         # Registro de operaciones
        self.extractor = FeatureExtractor()
        self.knn = KNNClassifier(k=3)
        self.kmeans = KMeansClustering(k=4)

    def log(self, message):
        self.logs.append(message)
        print(message)  # También imprime en consola para debug

    def load_dataset(self, directory):
        """
        Carga el dataset asumiendo que 'directory' contiene subcarpetas, 
        cada una con imágenes de una clase (por ejemplo: "tornillos", "clavos", etc.)
        """
        allowed_extensions = [".jpg", ".png", ".jpeg", ".bmp"]
        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in allowed_extensions:
                        img_path = os.path.join(class_path, file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            self.images.append((img, class_name))
        self.log("Dataset cargado. Total imágenes: {}".format(len(self.images)))

    def extract_all_features(self):
        """
        Extrae características de todas las imágenes del dataset.
        Cada vector de características tiene: [circularidad, aspect ratio, excentricidad, hu1]
        """
        feats = []
        labs = []
        for (img, label) in self.images:
            feat = self.extractor.extract_features(img)
            feats.append(feat)
            labs.append(label)
        self.features = np.array(feats)
        self.labels = np.array(labs)
        self.log("Extracción de características completada. Formato: [circularidad, aspect ratio, excentricidad, hu1]")

    def run_knn(self, test_image):
        test_feat = self.extractor.extract_features(test_image)
        prediction = self.knn.predict(self.features, self.labels, test_feat)
        self.log("KNN predicción: " + str(prediction))
        return prediction

    def run_kmeans(self):
        cluster_labels, centroids = self.kmeans.fit(self.features)
        self.log("K-means clustering completado.")
        return cluster_labels, centroids

# -------------------- VISTA --------------------

class View(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard - Trabajo Final IA I")
        self.setGeometry(100, 100, 1200, 800)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # Botones y etiquetas
        self.loadDataBtn = QPushButton("Cargar Dataset")
        self.extractBtn = QPushButton("Extraer Características")
        self.runKNNBtn = QPushButton("Ejecutar KNN")
        self.runKMeansBtn = QPushButton("Ejecutar K-means")
        self.predictionLabel = QLabel("Predicción: ")
        self.logTextEdit = QTextEdit()
        self.logTextEdit.setReadOnly(True)

        # Área para gráficos con matplotlib (canvas embebido)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)

        # Organizando la interfaz
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.loadDataBtn)
        btn_layout.addWidget(self.extractBtn)
        btn_layout.addWidget(self.runKNNBtn)
        btn_layout.addWidget(self.runKMeansBtn)
        
        self.layout.addLayout(btn_layout)
        self.layout.addWidget(self.predictionLabel)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(QLabel("Registro de Operaciones:"))
        self.layout.addWidget(self.logTextEdit)

# -------------------- CONTROLADOR --------------------

class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        # Conectar señales con slots
        self.view.loadDataBtn.clicked.connect(self.load_data)
        self.view.extractBtn.clicked.connect(self.extract_features)
        self.view.runKNNBtn.clicked.connect(self.run_knn)
        self.view.runKMeansBtn.clicked.connect(self.run_kmeans)

    def update_log(self):
        self.view.logTextEdit.setText("\n".join(self.model.logs))

    def load_data(self):
        directory = QFileDialog.getExistingDirectory(self.view, "Seleccionar carpeta de dataset")
        if directory:
            self.model.load_dataset(directory)
            self.model.log("Dataset cargado desde: " + directory)
            self.update_log()

    def extract_features(self):
        if not self.model.images:
            self.model.log("¡Oye! Primero carga el dataset.")
        else:
            self.model.extract_all_features()
        self.update_log()

    def run_knn(self):
        # Permitir al usuario seleccionar una imagen de prueba
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self.view, "Seleccionar imagen de prueba", "",
                                                  "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if fileName:
            test_image = cv2.imread(fileName)
            if test_image is None:
                self.model.log("Error al cargar la imagen de prueba.")
            else:
                prediction = self.model.run_knn(test_image)
                self.view.predictionLabel.setText("Predicción KNN: " + str(prediction))
        self.update_log()

    def run_kmeans(self):
        if self.model.features is None:
            self.model.log("Primero extrae las características del dataset.")
        else:
            cluster_labels, centroids = self.model.run_kmeans()
            # Crear gráfico 3D con las características: 
            # Eje X: circularidad, Eje Y: aspect ratio, Eje Z: Hu Moment 1
            self.view.figure.clear()
            ax = self.view.figure.add_subplot(111, projection='3d')
            xs = self.model.features[:, 0]
            ys = self.model.features[:, 1]
            zs = self.model.features[:, 3]
            sc = ax.scatter(xs, ys, zs, c=cluster_labels, cmap='viridis', s=50)
            ax.set_xlabel("Circularidad")
            ax.set_ylabel("Aspect Ratio")
            ax.set_zlabel("Hu Moment 1")
            ax.set_title("Clustering K-means")
            self.view.figure.colorbar(sc, ax=ax, shrink=0.5)
            self.view.canvas.draw()
            self.model.log("Gráfico de K-means actualizado.")
        self.update_log()

# -------------------- MAIN --------------------

def main():
    app = QApplication(sys.argv)
    model = Model()
    view = View()
    controller = Controller(model, view)
    view.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()