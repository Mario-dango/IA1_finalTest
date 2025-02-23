import cv2
import numpy as np
from skimage.measure import moments, moments_hu

class ExtractorCaracteristicas:
    def __init__(self, imagen):
        """
        Constructor de la clase.
        :param imagen: Imagen de entrada en formato BGR (OpenCV).
        """
        self.imagen = imagen  # Almacena la imagen original
        self.gris = None  # Almacenará la imagen en escala de grises
        self.binaria = None  # Almacenará la imagen binaria
        self.contorno = None  # Almacenará el contorno del objeto
        self.area = None  # Almacenará el área del contorno
        self.perimetro = None  # Almacenará el perímetro del contorno
        self.momentos_hu = None  # Almacenará los momentos de Hu

    def preprocesar_imagen(self):
        """
        Convierte la imagen a escala de grises y la binariza.
        """
        # Convertir la imagen a escala de grises
        self.gris = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        # Aplicar umbralización para obtener una imagen binaria
        _, self.binaria = cv2.threshold(self.gris, 127, 255, cv2.THRESH_BINARY_INV)

    def encontrar_contorno(self):
        """
        Encuentra el contorno del objeto en la imagen binaria.
        """
        # Encontrar contornos en la imagen binaria
        contornos, _ = cv2.findContours(self.binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Seleccionar el contorno más grande (objeto de interés)
        self.contorno = max(contornos, key=cv2.contourArea)

    def calcular_area_perimetro(self):
        """
        Calcula el área y el perímetro del contorno.
        """
        # Calcular el área del contorno
        self.area = cv2.contourArea(self.contorno)
        # Calcular el perímetro del contorno
        self.perimetro = cv2.arcLength(self.contorno, True)

    def calcular_excentricidad(self, mu):
        """
        Calcula la excentricidad a partir de los momentos centrales.
        :param mu: Momentos centrales de la imagen.
        :return: Excentricidad del objeto.
        """
        if mu[2, 0] + mu[0, 2] == 0:
            return 0
        numerador = (mu[2, 0] - mu[0, 2])**2 + 4 * mu[1, 1]**2
        denominador = (mu[2, 0] + mu[0, 2])**2
        return np.sqrt(numerador / denominador)

    def calcular_circularidad(self):
        """
        Calcula la circularidad del objeto.
        :return: Circularidad del objeto.
        """
        if self.perimetro == 0:
            return 0
        return (4 * np.pi * self.area) / (self.perimetro**2)

    def calcular_aspect_ratio(self):
        """
        Calcula el aspect ratio (relación de aspecto) del objeto.
        :return: Aspect ratio del objeto.
        """
        # Obtener el rectángulo delimitador del contorno
        x, y, w, h = cv2.boundingRect(self.contorno)
        return float(w) / h

    def calcular_momentos_hu(self):
        """
        Calcula los momentos de Hu de la imagen.
        """
        # Calcular los momentos de la imagen en escala de grises
        M = moments(self.gris)
        # Calcular los momentos de Hu
        self.momentos_hu = moments_hu(M)

    def extraer_caracteristicas(self):
        """
        Extrae todas las características de la imagen.
        :return: Diccionario con las características ordenadas.
        """
        # Preprocesar la imagen (escala de grises y binarización)
        self.preprocesar_imagen()
        # Encontrar el contorno del objeto
        self.encontrar_contorno()
        # Calcular el área y el perímetro del contorno
        self.calcular_area_perimetro()
        # Calcular los momentos de Hu
        self.calcular_momentos_hu()

        # Calcular las características
        excentricidad = self.calcular_excentricidad(moments(self.gris))
        circularidad = self.calcular_circularidad()
        aspect_ratio = self.calcular_aspect_ratio()

        # Devolver las características en un diccionario ordenado
        return {
            'excentricidad': excentricidad,
            'circularidad': circularidad,
            'aspect_ratio': aspect_ratio,
            'momentos_hu': self.momentos_hu.tolist()  # Convertir a lista para serialización
        }


# # Ejemplo de uso
# if __name__ == "__main__":
#     # Cargar la imagen
#     imagen = cv2.imread('ruta/a/tu/imagen.jpg')
    
#     # Crear una instancia de la clase ExtractorCaracteristicas
#     extractor = ExtractorCaracteristicas(imagen)
    
#     # Extraer características
#     caracteristicas = extractor.extraer_caracteristicas()
    
#     # Mostrar las características
#     print("Características extraídas:")
#     for key, value in caracteristicas.items():
#         print(f"{key}: {value}")