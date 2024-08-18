import cv2

#Defino la ruta de la imagen
# pathImage = "../resources/images/formasP.png"
pathImage = "../resources/pictures/tornillo01.jpeg"
# pathImage = "../resources/pictures/tuerca01.jpeg"

pathImage = "../resources/dataset/internet/arandela01.jpg" #No
pathImage = "../resources/dataset/internet/arandela02.jpg"
pathImage = "../resources/dataset/internet/arandela03.jpg"
pathImage = "../resources/dataset/internet/arandela04.jpg"
pathImage = "../resources/dataset/internet/arandela05.jpg" #No
pathImage = "../resources/dataset/internet/arandela06.jpg" #No

pathImage = "../resources/dataset/internet/tornillo01.jpg" #No
pathImage = "../resources/dataset/internet/tornillo02.jpg"
pathImage = "../resources/dataset/internet/tornillo03.jpg" #No
pathImage = "../resources/dataset/internet/tornillo04.jpg"
pathImage = "../resources/dataset/internet/tornillo05.jpg"
pathImage = "../resources/dataset/internet/tornillo06.jpg" #No

pathImage = "../resources/dataset/internet/tuerca01.jpg"
pathImage = "../resources/dataset/internet/tuerca02.jpg"
pathImage = "../resources/dataset/internet/tuerca03.jpg" #No
pathImage = "../resources/dataset/internet/tuerca04.jpg" #No
pathImage = "../resources/dataset/internet/tuerca05.jpg"
pathImage = "../resources/dataset/internet/tuerca06.jpg"

pathImage = "../resources/dataset/internet/clavo01.jpg" #No
pathImage = "../resources/dataset/internet/clavo02.jpg"
pathImage = "../resources/dataset/internet/clavo03.jpg"
pathImage = "../resources/dataset/internet/clavo04.jpg" #No
pathImage = "../resources/dataset/internet/clavo05.jpg" #No
pathImage = "../resources/dataset/internet/clavo06.jpg"

pathImage = "../resources/images/formas1.png"
# # pathImage = "../resources/images/tu.png"
# pathImage = "../resources/images/ar.png"

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


### Canny edge Detection funtion ###
def getCanny(img):
    img = resize(img, width = 300, height = 300)
    # Setting parameter values 
    t_lower = 80  # Lower Threshold, bajo de este nivel no detecta el contorno.
    t_upper = 100  # Upper threshold, entre lower y upper se dibuja solo si el contorno conecta con uno de valor mayor a upper
    aperture_size = 3  # Aperture size 
    L2Gradient = False # Boolean 
    img = cv2.GaussianBlur(img, (5,7), 0)
    imgCanny = cv2.Canny(img, t_lower, t_upper, apertureSize = aperture_size,  L2gradient = L2Gradient)
    # Encuentra los contornos en la imagen filtrada por Canny
    contC,_ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"El valor de contorno de Canny es: \n {contC}")
    cv2.imshow("imagen filtro Canny", imgCanny)
    cv2.drawContours(imgCanny, contC, 0, (255, 255, 255), 3)
    cv2.imshow("imagen filtro Canny c contorno", imgCanny)
    return imgCanny, contC



img = cv2.imread(pathImage)
imgR = resize(img, 300, 300)
imgC, cont = getCanny(img)
print(type(cont))
print("la cantidad de contornos es de:" + str(len(cont)))
cv2.drawContours(imgR, cont, 0, (0, 255, 47), 3)
cv2.imshow("Imagen original",imgR)



cv2.waitKey(0)
cv2.destroyAllWindows()

