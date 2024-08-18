import cv2 
import numpy as np
import matplotlib as plt
from numpy.fft import fft2, fftshift, ifft2 

# contC = []
# imgCanny = 0
# contS = []
# imgSobel = 0
# ContG = []
# imgGauss = 0

def procesar(imagen): #funcion para procesar la imagen (con filtros)
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

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Función que cambia el tamaño de una imagen preservando la relación de aspecto.
    :param image: Imagen a ser alterada.
    :param width: Ancho objetivo (opcional).
    :param height: Altura objetivo (opcional).
    :param inter: Método de interpolación (por defecto: cv2.INTER_AREA)
    :return: Imagen redimensionada. Se le da prioridad a *height*, por lo que si se especifican tanto *width*
             como *height*, *width* será ignorado.
    """
    # Extraemos las dimensiones originales.
    (original_height, original_width) = image.shape[:2]
    # Si no se especifica al menos uno de los parámetros, no tenemos nada que hacer aparte de retornar.
    if width is None and height is None:
        return image
    # Si el nuevo ancho es vacío (*width*), calcularemos la relación de aspecto con base a la nueva altura (*height*)
    if width is None:
        # Proporción para mantener la relación de aspecto con base a la nueva altura.
        ratio = height / float(original_height)
        # Nueva anchura
        width = int(original_width * ratio)
    else:
        # Proporción para mantener la relación de aspecto con base a la nueva anchura.
        ratio = width / float(original_width)
        # Nueva altura
        height = int(original_height * ratio)
    # El nuevo tamaño de la imagen no será más que un par compuesta por la nueva anchura y la nueva altura.
    new_size = (width, height)
    # Usamos la función cv2.resize() para llevar a cabo el cambio de tamaño de la imagen; finalmente retornamos el
    # resultado.
    return cv2.resize(image, new_size, interpolation=inter)

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def getGaussSobel(img):    
    img = resize(img, width = 300, height = 300)
    t_lower = 150  # Lower Threshold 
    t_upper = 250  # Upper threshold 
    blurred = cv2.GaussianBlur(img, (9, 9), 0)     
    #Creo una imagen apartir de una transformación a grices de la imagen original
    grices = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _,imgGauss = cv2.threshold(grices, t_lower, t_upper, cv2.THRESH_BINARY_INV)
    #Consigo los contornos ordenandolos en cantidad y por jerarquia
    contG,_ = cv2.findContours(imgGauss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"El valor de contorno de Gauss es: \n {contG}")  
    cv2.imshow("Blur Gauss + Sobel ", imgGauss)
    return imgGauss

### Filtro Sobel
def getSobel(img):    
    # img = resize(img, width = 300, height = 300)
    # t_lower = 70  # Lower Threshold 
    # t_upper = 200  # Upper threshold     
    # #Creo una imagen apartir de una transformación a grices de la imagen original
    # grices = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _,imgSobel = cv2.threshold(grices, t_lower, t_upper, cv2.THRESH_BINARY_INV)
    # #Consigo los contornos ordenandolos en cantidad y por jerarquia
    # contS,_ = cv2.findContours(imgSobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # print(f"El valor de contorno de Sobel es: \n {contS}")
    # #dibujo los contornos
    # cv2.imshow("Imagen sobel", imgSobel)
    # cv2.drawContours(imagen, contS, 0, (0,255,0),3)
    # cv2.imshow("Imagen sobel Contorneada", imagen)
    # return imgSobel
    pass

### Canny edge Detection funtion ###
def getCanny(img):
    img = resize(img, width = 300, height = 300)
    # Setting parameter values 
    t_lower = 75  # Lower Threshold 
    t_upper = 150  # Upper threshold 
    aperture_size = 3  # Aperture size 
    L2Gradient = False # Boolean 
    img = cv2.GaussianBlur(img, (11,11), 0)
    imgCanny = cv2.Canny(img, t_lower, t_upper, apertureSize = aperture_size,  L2gradient = L2Gradient)
    # Encuentra los contornos en la imagen filtrada por Canny
    contC,_ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"El valor de contorno de Canny es: \n {contC}")
    cv2.imshow("imagen filtro Canny", imgCanny)
    cv2.drawContours(imgCanny, contC, 0, (255, 255, 255), 3)
    cv2.imshow("imagen filtro Canny c contorno", imgCanny)
    return imgCanny


#Defino la ruta de la imagen
# pathImage = "../resources/images/formasP.png"
# pathImage = "../resources/pictures/tornillo01.jpeg"
# pathImage = "../resources/pictures/tuerca01.jpeg"
# pathImage = "../resources/dataset/internet/arandela06.jpg"
# # pathImage = "../resources/images/formas1.png"
# # pathImage = "../resources/images/tu.png"
# pathImage = "../resources/images/ar.png"
# # pathImage = "../resources/dataset/internet/tornillo04.jpg"
pathImage = "../resources/dataset/internet/clavo06.jpg"


###### TUERCA #####
    # Hu Moments de la imagen filtrada por Canny:
    # Hu Moment 0: [0.16043255]
    # Hu Moment 1: [2.1470075e-05]
    # Hu Moment 2: [5.64115258e-07]
    # Hu Moment 3: [5.76514126e-11]
    # Hu Moment 4: [-1.12803369e-19]
    # Hu Moment 5: [-1.76310183e-13]
    # Hu Moment 6: [-3.088178e-19]
    # El valor de elipse es:  ((145.84495544433594, 149.1978302001953), (186.51580810546875, 187.3463134765625), 97.8422622680664)
    # Excentricidad: 1.004452734486849
    # 635.3279880285263

##### ARANDELA #####
    # Hu Moments de la imagen filtrada por Canny:
    # Hu Moment 0: [0.15967125]
    # Hu Moment 1: [0.00016231]
    # Hu Moment 2: [2.20245049e-07]
    # Hu Moment 3: [3.14492684e-10]
    # Hu Moment 4: [-7.29073564e-19]
    # Hu Moment 5: [-1.3926878e-12]
    # Hu Moment 6: [-2.51380124e-18]
    # El valor de elipse es:  ((143.64830017089844, 146.52069091796875), (221.28506469726562, 239.99610900878906), 179.875732421875)
    # Excentricidad: 1.084556290941377
    # 764.6660826206207

##### TORNILLO ######
    # Hu Moments de la imagen filtrada por Canny:
    # Hu Moment 0: [0.59763315]
    # Hu Moment 1: [0.31755715]
    # Hu Moment 2: [0.00666327]
    # Hu Moment 3: [0.00312985]
    # Hu Moment 4: [1.42827528e-05]
    # Hu Moment 5: [0.00175945]
    # Hu Moment 6: [-5.46599259e-07]
    # El valor de elipse es:  ((140.4454803466797, 82.31503295898438), (44.475677490234375, 353.2896423339844), 179.60968017578125)
    # Excentricidad: 7.94343475513233
    # 642.8254653215408

###### CLAVO #####
    # Hu Moments de la imagen filtrada por Canny:
    # Hu Moment 0: [0.7531049]
    # Hu Moment 1: [0.51903261]
    # Hu Moment 2: [0.01598998]
    # Hu Moment 3: [0.00783349]
    # Hu Moment 4: [8.76699967e-05]
    # Hu Moment 5: [0.00564338]
    # Hu Moment 6: [-4.41385088e-07]
    # El valor de elipse es:  ((142.9859619140625, 151.37010192871094), (167.90377807617188, 204.0690155029297), 174.81268310546875)
    # Excentricidad: 1.2153926364322245
    # 535.8406196832657

#Leo la imagen
imagen = cv2.imread(pathImage)
imagen = resize(imagen, width = 300, height = 300)
cv2.imshow("imagen real sin contorno", imagen)


# imageGauss = cv2.GaussianBlur(imagen, (5, 5), 0)
# _,asdasad = cv2.threshold(imageGauss, 78, 150, cv2.THRESH_BINARY_INV)
imgCanny = getCanny(imagen)
imgGauss = getGaussSobel(imagen)
imgSobel = getSobel(imagen) 
cv2.waitKey(0)


grices = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
_,imgSobel = cv2.threshold(grices, 130, 220, cv2.THRESH_BINARY_INV)
#Consigo los contornos ordenandolos en cantidad y por jerarquia
contS,_ = cv2.findContours(imgSobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

imgCanny = cv2.Canny(imagen, 130, 220)
# Encuentra los contornos en la imagen filtrada por Canny
contC,_ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contC
print(type(contours))
cv2.waitKey(0)
cv2.dft()

# ## prueba de función Lucho
# procesar(imagen)

# # Calcula los momentos de Hu para el primer contorno encontrado
# for c in range(len(contours)):
#     moments = cv2.moments(contours[c])
#     hu_moments = cv2.HuMoments(moments)
#     print("Hu Moments de la imagen filtrada por Canny:")
#     for i in range(0, 7):
#         print(f"Hu Moment {i}: {hu_moments[i]}")

# #por cada elemento dentro de  la cantidad de contornosdasd
# for conts in range (len(contC)):     
#     #trabajo con el contorno i-ésimo
#     cnt = contC[conts]   
#     #consigo su momento de hu
#     M = cv2.moments(cnt)
#     #muestro su valor
#     # print(M)
#     #Calculo sus coordenadas del centroide en X y en Y
#     if M["m00"] != 0:
#         cX = int (M["m10"]/M["m00"]);cY = int (M["m01"]/M["m00"])
#         #imprimo sus valores
#         # print(cX);print(cY)
#         #Dibujo un punto en donde corresponde la coordenada de su centroide
#         cv2.circle(imagen,(cX,cY),5,(130,130,130),-1)
#     #elipse
#     elip = 0; excentricity = 0
#     if len(contC[conts]) > 5:            
#         elip = cv2.fitEllipse(contC[conts])
#         print("El valor de elipse es: ", elip)
#         # Calcular los ejes mayor y menor de la elipse ajustada
#         major_axis = max(elip[1])
#         minor_axis = min(elip[1])
#         # Calcular la excentricidad
#         excentricity = major_axis / minor_axis

#         # Imprimir el valor de excentricidad
#         print("Excentricidad:", excentricity)
        
    
#     #Calculo la longitud de curva del contorno
#     per = cv2.arcLength(cnt, True)
#     #imprimo la longitud de curva del contorno
#     print(per)
#     #Aproximo las curvas de contorno por medio de rectas
#     approx = cv2.approxPolyDP(cnt, 0.006*per, True)
#     #imprimo 
#     print(approx);print(len(approx))
    
    
#     # Calcular la cantidad de lados
#     num_lados = len(approx)
#     longitud = []
#     # Calcular la longitud de cada lado
#     for i in range(num_lados):
#         # Calcular la distancia entre puntos adyacentes para obtener la longitud del lado
#         p1 = tuple(approx[i][0])  # Punto actual
#         p2 = tuple(approx[(i + 1) % num_lados][0])  # Punto siguiente (teniendo en cuenta el contorno cerrado)

#         # Calcular la distancia euclidiana entre los puntos
#         longitud_lado = np.linalg.norm(np.array(p2) - np.array(p1))
#         longitud.append(longitud_lado)
#         # Imprimir la longitud de cada lado
#         print(f"Longitud del lado {i + 1}: {longitud_lado}")

#     # Imprimir la cantidad de lados
#     print(f"Cantidad de lados: {num_lados}")
#     mediana = np.median(longitud)
#     media = np.mean(longitud)
#     print(f"El valor de mediana es {mediana} y el de media es {media}")
#     print(f"su exentricidad es de {excentricity}")
    
    
#     #Muestro las ventanas de las imagenes
#     # cv2.imshow("Imagen real", imagen);cv2.imshow("th",getSobel(imagen))
#     #Espero presionar una tecla para pasar a la siguiente iteración
#     cv2.waitKey(0)

# #limprio las ventanas existentes
# cv2.destroyAllWindows()