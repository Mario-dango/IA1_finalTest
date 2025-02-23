import math
import cv2
import numpy as np

class ExtractorDeCaracteristicas:
    def __init__(self, imagen_binaria):
        self.imagen = imagen_binaria

    def calcular_caracteristicas(self):
        # Encontrar los contornos
        contornos, _ = cv2.findContours(self.imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Asegurarse de que se encontró al menos un contorno
        if len(contornos) == 0:
            return None

        # Tomar el contorno más grande (asumiendo que es el objeto de interés)
        contorno = max(contornos, key=cv2.contourArea)

        # Calcular el área
        area = cv2.contourArea(contorno)

        # Calcular el perímetro
        perimetro = cv2.arcLength(contorno, True)

        # Calcular la circularidad
        circularidad = 4 * np.pi * area / (perimetro * perimetro)

        # Calcular los momentos de Hu
        momentos = cv2.moments(contorno)
        hu_momentos = cv2.HuMoments(momentos)

        # Deshacer la normalización de los momentos de Hu
        for i in range(0, 7):
            hu_momentos[i] = -1 * hu_momentos[i] * math.copysign(1.0, hu_momentos[i])

        return {
            'area': area,
            'perimetro': perimetro,
            'circularidad': circularidad,
            'momentos_hu': hu_momentos
        }