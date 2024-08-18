import os
import cv2

class MainController:
    def __init__(self):
        pass
    
    def procesar(self, imagen): #funcion para procesar la imagen (con filtros)
        imagen = cv2.resize(imagen, None, fx=0.15, fy=0.15)
        imagen = cv2.GaussianBlur(imagen, (7,7), 0)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen = cv2.Canny(imagen,100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        imagen = cv2.dilate(imagen, kernel, iterations=1)
        contornos, jerarquia = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imagen, contornos, -1, (255,255,255), -1)
        imagen = cv2.erode(imagen, kernel, iterations=3)
        return imagen
    
    def obtener_momentos(self):
        momentos_hu = []
        clases = []
        path_images = "../resources/dataset/internet/"
        for carpeta in os.listdir(path_images):
            for imagen in os.listdir(path_images + carpeta):
                img = cv2.imread(path_images + carpeta + '/' + imagen)
                img_proc = self.procesar(img)
                momentos = cv2.moments(img_proc)
                hu_moments = cv2.HuMoments(momentos)
                momentos_hu.append(hu_moments)
                clases.append(carpeta)
        return momentos_hu, clases