import cv2
import os
import numpy as np
from model.image_model import ImageModel  # Importar ImageModel

class DatabaseModel:
    def __init__(self):
        """
        Inicializa el modelo de la base de datos.
        """
        self.datos_entrenamiento = []  # Almacena las características y etiquetas de las imágenes
        self.mean = None  # Media para normalización
        self.std = None   # Desviación estándar para normalización
        self.image_model = ImageModel()  # Instancia de ImageModel para calcular características

    def cargar_dataset(self, ruta_dataset):
        """
        Carga un dataset desde una carpeta. Se espera que dentro de 'ruta_dataset'
        existan subcarpetas con el nombre de cada clase, y que cada subcarpeta contenga imágenes.
        Retorna el número de imágenes cargadas.
        """
        self.datos_entrenamiento = []  # Limpiar dataset anterior
        extensiones = ('.png', '.jpg', '.jpeg', '.bmp')  # Extensiones de imágenes soportadas

        # Recorrer las subcarpetas (cada una representa una clase)
        for subfolder in os.listdir(ruta_dataset):
            ruta_sub = os.path.join(ruta_dataset, subfolder)
            if os.path.isdir(ruta_sub):
                # Recorrer las imágenes en la subcarpeta
                for archivo in os.listdir(ruta_sub):
                    if archivo.lower().endswith(extensiones):
                        ruta_img = os.path.join(ruta_sub, archivo)
                        imagen = cv2.imread(ruta_img)
                        if imagen is not None:
                            # Calcular características de la imagen usando ImageModel
                            circ, asp, exc, hu0 = self.image_model.calcular_caracteristicas(imagen)
                            if circ != 0 or asp != 0:  # Filtrar imágenes no válidas
                                self.datos_entrenamiento.append([circ, asp, hu0, subfolder])

        # Actualizar normalización
        self.actualizar_normalizacion()
        return len(self.datos_entrenamiento)

    def actualizar_normalizacion(self):
        """
        Calcula la media y la desviación estándar de las características en el dataset de entrenamiento.
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

    def desnormalizar(self, features):
        """
        Des-normaliza un vector de características (inverso de normalizar).
        """
        if self.mean is None or self.std is None:
            return features
        return features * self.std + self.mean