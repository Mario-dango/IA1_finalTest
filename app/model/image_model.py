import cv2
import math
import numpy as np

class ImageModel:
    def calcular_caracteristicas(self, imagen):
        """
        Dada una imagen (BGR), retorna los 3 momentos de Hu principales (h0, h1, h6)
        transformados para una mejor escala.
        """        
        imagen_filtrada = self.filtrado(imagen)
        contornos, _ = cv2.findContours(imagen_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contornos:
            return 0, 0, 0

        contorno = max(contornos, key=cv2.contourArea)

        # Calcular los 7 momentos de Hu
        momentos = cv2.moments(contorno)
        hu = cv2.HuMoments(momentos)

        # Función para transformar los momentos de Hu, evitando valores de 0.
        def transformar_hu(valor):
            if valor == 0:
                return 0.0
            return -np.sign(valor) * np.log10(abs(valor))

        # Extraer y transformar h0, h1 y h6.
        hu0 = transformar_hu(hu[0][0])
        # hu1 = transformar_hu(hu[1][0])
        # hu6 = transformar_hu(hu[6][0])

        # 'contour' es el contorno del objeto que estás analizando
        area = cv2.contourArea(contorno)
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)

        # Evitar división por cero
        if hull_area > 0:
            solidity = float(area) / hull_area
        else:
            solidity = 0
        # Si la solidez es mayor a 0.95 es probable que sea un clavo
        
        # 'contour' es el contorno del objeto
        perimeter = cv2.arcLength(contorno, True)
        area = cv2.contourArea(contorno)

        # Evitar división por cero
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter**2)
        else:
            circularity = 0

        # Resumen de la Estrategia de Clasificación
        # Con estas tres características, tu árbol de decisión para clasificar sería:

        # Calcular h0:
        # Si h0 es alto (objeto redondo): Pasa al paso 2.
        # Si h0 es bajo (objeto alargado): Pasa al paso 3.

        # Calcular Circularidad (circularity):
        # Si circularity > 0.9 (aprox.): Es una Arandela.
        # Si circularity < 0.9 (aprox.): Es una Tuerca.

        # Calcular Solidez (solidity):
        # Si solidity > 0.95 (aprox.): Es un Clavo.
        # Si solidity < 0.95 (aprox.): Es un Tornillo.

        
        return hu0, solidity, circularity

    def generar_imagen_contorno(self, imagen_cv):
        # Obtener la imagen con bordes/umbrales
        imagen_filtrada = self.filtrado(imagen_cv)
        # Buscar contornos
        contornos, _ = cv2.findContours(imagen_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Copiar la imagen original para dibujar sobre ella
        contorno_img = self.resize(imagen_cv.copy(), 512, 512)
        
        if contornos:
            # Seleccionar el contorno de mayor área
            contorno_principal = max(contornos, key=cv2.contourArea)
            cv2.drawContours(contorno_img, [contorno_principal], -1, (0, 255, 0), 2)
        
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

    def filtrado(self, img):
        # Opcional: redimensionar (ajusta según tu caso)
        img = self.resize(img, width=512, height=512)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
        # cv2.imshow("img_gris",gray)
        # Aplicar un filtro de mediana o bilateral
        gray = cv2.medianBlur(gray, 5)       
        # cv2.imshow("img_grisBlur",gray)
        # gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Umbralizar la imagen (Otsu o adaptativo) para resaltar el objeto
        # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # -o- adaptativo:
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5
        )       
        # cv2.imshow("img_thresh",thresh)
        
        # Operaciones morfológicas
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Canny (con umbrales adaptados a la mediana)
        median_val = np.median(gray)
        lower = int(max(0, (1.0 - 0.33) * median_val))
        upper = int(min(255, (1.0 + 0.33) * median_val))
        edges = cv2.Canny(thresh, lower, upper)
        # cv2.imshow("img_Canny",edges)
        
        # Cierre final si hace falta
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        # cv2.imshow("img_morphClos",edges)
        
        return edges



    def mofologicTransform(self, img, operation,iterations = 2):
        # Defino la matriz que afectará a los pixeles morfologicamente
        sizeKernel = np.ones((4,3), np.uint8)
        iterations = 1
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