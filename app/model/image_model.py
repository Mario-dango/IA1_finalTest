import cv2
import math
import numpy as np

class ImageModel:
    def calcular_caracteristicas(self, imagen):
        """
        Dada una imagen (BGR), retorna:
         - Circularidad
         - Aspect Ratio
         - Excentricidad (calculada pero no usada en la clasificación 3D)
         - Primer momento de Hu (hu0)
        """
        # # Convertir a escala de grises
        # gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # # Suavizado para reducir ruido y sombras
        # blur = cv2.GaussianBlur(gray, (7, 7), 0)
        # # Threshold adaptativo
        # blockSize = 21  # Tamaño de la ventana (debe ser impar)
        # C = 5           # Constante para ajustar el umbral
        # thresh = cv2.adaptiveThreshold(
        #     blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C
        # )
        # # Operación morfológica de cierre para eliminar pequeños huecos
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        imagen_filtrada = self.filtrado(imagen)
        # Buscar contornos
        contornos, _ = cv2.findContours(imagen_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    def generar_imagen_contorno(self, imagen_cv):
        """
        Genera una imagen con los contornos resaltados.
        """
        # gray = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY_INV, 11, 2)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        imagen_filtrada = self.filtrado(imagen_cv)
        contornos, _ = cv2.findContours(imagen_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno_img = imagen_cv.copy()
        cv2.drawContours(contorno_img, contornos, 0, (0, 255, 47), 3)
        return contorno_img
    
    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
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
    def filtrado(self, img):
        img = self.resize(img, width = 300, height = 300)
        # # Setting parameter values 
        # t_lower = 50  # Lower Threshold, bajo de este nivel no detecta el contorno.
        # t_upper = 160  # Upper threshold, entre lower y upper se dibuja solo si el contorno conecta con uno de valor mayor a upper
        
        t_lower = 100  # Lower Threshold, bajo de este nivel no detecta el contorno.
        t_upper = 200  # Upper threshold, entre lower y upper se dibuja solo si el contorno conecta con uno de valor mayor a upper
        aperture_size = 3  # Aperture size 
        L2Gradient = False # Boolean 
        img_gauss = cv2.GaussianBlur(img, (5,5), 0)
        imgCanny = cv2.Canny(img_gauss, t_lower, t_upper, apertureSize = aperture_size,  L2gradient = L2Gradient)
        # Encuentra los contornos en la imagen filtrada por Canny
        contC,_ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"\nEl valor de contorno de Canny es: \n {contC}")
        return imgCanny

    def mofologicTransform(self, img, operation,iterations = 2):
        # Defino la matriz que afectará a los pixeles morfologicamente
        sizeKernel = np.ones((4,3), np.uint8)
        if operation == "erosion":
            # Transformación morfologica de erosión
            operacion = cv2.erode(img, sizeKernel, iterations)
        elif operation == "dilatacion":
            # Transformación morfologica de erosión
            operacion = cv2.dilate(img, sizeKernel, iterations)
        elif operation == "opening":
            # Transformación morfologica de erosión
            operacion = cv2.morphologyEx(img, cv2.MORPH_OPEN, sizeKernel, iterations)
        elif operation == "closing":
            # Transformación morfologica de erosión
            operacion = cv2.morphologyEx(img, cv2.MORPH_CLOSE, sizeKernel, iterations)
        else:
            print("operacion invaldiad")
            operacion = None
        return operacion