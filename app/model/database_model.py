import cv2                      # Importa OpenCV para cargar y procesar imágenes
import os                       # Importa os para operaciones con archivos y carpetas
import numpy as np              # Importa NumPy para manejo de arrays y cálculos numéricos
from model.image_model import ImageModel  # Importa ImageModel para extraer características de las imágenes

class DatabaseModel:
    def __init__(self):
        """
        Inicializa el modelo de la base de datos.
        """
        self.datos_entrenamiento = []  # Lista para almacenar características y etiquetas de cada imagen
        self.mean = None               # Media de las características para normalización
        self.std = None                # Desviación estándar para normalización
        self.image_model = ImageModel()  # Instancia de ImageModel para calcular características

    def cargar_dataset(self, ruta_dataset):
        """
        Carga un dataset desde una carpeta. Se espera que dentro de 'ruta_dataset'
        existan subcarpetas con el nombre de cada clase, y cada subcarpeta contenga imágenes.
        Retorna el número de imágenes cargadas.
        :param ruta_dataset: Ruta a la carpeta del dataset.
        """
        self.datos_entrenamiento = []  # Reinicia la lista de datos de entrenamiento
        extensiones = ('.png', '.jpg', '.jpeg', '.bmp')  # Extensiones de imágenes soportadas

        # Recorre cada subcarpeta en la carpeta del dataset
        for subfolder in os.listdir(ruta_dataset):
            ruta_sub = os.path.join(ruta_dataset, subfolder)
            if os.path.isdir(ruta_sub):
                # Recorre cada archivo en la subcarpeta
                for archivo in os.listdir(ruta_sub):
                    if archivo.lower().endswith(extensiones):
                        ruta_img = os.path.join(ruta_sub, archivo)
                        imagen = cv2.imread(ruta_img)  # Lee la imagen usando OpenCV
                        if imagen is not None:
                            # Calcula las características de la imagen con ImageModel
                            circ, asp, exc, hu0 = self.image_model.calcular_caracteristicas(imagen)
                            # Solo se añaden imágenes válidas (al menos una característica distinta de cero)
                            if circ != 0 or asp != 0:
                                self.datos_entrenamiento.append([circ, asp, hu0, subfolder])

        # Una vez cargado el dataset, actualiza la normalización
        self.actualizar_normalizacion()
        return len(self.datos_entrenamiento)

    def actualizar_normalizacion(self):
        """
        Calcula la media y la desviación estándar de las primeras 3 características
        (usadas en la normalización) a partir de los datos de entrenamiento.
        """
        if not self.datos_entrenamiento:
            return
        # Extrae las primeras 3 características de cada muestra y las convierte en array NumPy
        datos = np.array([fila[:3] for fila in self.datos_entrenamiento], dtype=np.float32)
        self.mean = np.mean(datos, axis=0)  # Calcula la media a lo largo de cada columna
        self.std = np.std(datos, axis=0)    # Calcula la desviación estándar
        self.std[self.std == 0] = 1           # Evita división por cero

    def normalizar(self, features):
        """
        Normaliza un vector de características usando la media y desviación estándar calculadas.
        :param features: Vector de características.
        :return: Vector normalizado.
        """
        if self.mean is None or self.std is None:
            return features
        return (np.array(features) - self.mean) / self.std

    def desnormalizar(self, features):
        """
        Realiza la operación inversa a la normalización, recuperando los valores originales.
        :param features: Vector normalizado.
        :return: Vector desnormalizado.
        """
        if self.mean is None or self.std is None:
            return features
        return features * self.std + self.mean
