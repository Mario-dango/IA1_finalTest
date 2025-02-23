import cv2
import numpy as np
import extract as ex
pathImage = "../resources/dataset/internet/arandelas/arandela04.jpg"

def getCanny(img):
    # img = resize(img, width = 300, height = 300)
    # Setting parameter values 
    t_lower = 50  # Lower Threshold, bajo de este nivel no detecta el contorno.
    t_upper = 200  # Upper threshold, entre lower y upper se dibuja solo si el contorno conecta con uno de valor mayor a upper
    
    # t_lower = 100  # Lower Threshold, bajo de este nivel no detecta el contorno.
    # t_upper = 200  # Upper threshold, entre lower y upper se dibuja solo si el contorno conecta con uno de valor mayor a upper
    aperture_size = 3  # Aperture size 
    L2Gradient = False # Boolean 
    img = cv2.GaussianBlur(img, (5,5), 0)
    imgCanny = cv2.Canny(img, t_lower, t_upper, apertureSize = aperture_size,  L2gradient = L2Gradient)
    # Encuentra los contornos en la imagen filtrada por Canny
    contC,_ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"El valor de contorno de Canny es: \n {contC}")
    return imgCanny, contC


# Suponiendo que tienes una imagen binaria cargada en una variable llamada 'imagen_binaria'
imagen_binaria = cv2.imread(pathImage)

imgC, cont = getCanny(imagen_binaria)
extractor = ex(imgC)
caracteristicas = extractor.calcular_caracteristicas()

if caracteristicas:
    print("Área:", caracteristicas['area'])
    print("Perímetro:", caracteristicas['perimetro'])
    print("Circularidad:", caracteristicas['circularidad'])
    print("Momentos de Hu:", caracteristicas['momentos_hu'])
else:
    print("No se encontró ningún contorno")