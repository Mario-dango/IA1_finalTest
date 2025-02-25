import cv2                     # Importa OpenCV para procesamiento de imágenes
import math                    # Importa math para funciones matemáticas (pi, sqrt, etc.)
import numpy as np             # Importa NumPy para manejo de arrays y cálculos numéricos

class ImageModel:
    def calcular_caracteristicas(self, imagen):
        """
        Dada una imagen en formato BGR, retorna:
         - Circularidad
         - Aspect Ratio (relación ancho/alto)
         - Excentricidad (calculada mediante ajuste de elipse; no se usa en clasificación 3D)
         - Primer momento de Hu (hu0)
        """
        # Se aplica el filtrado (detección de bordes) a la imagen
        imagen_filtrada = self.filtrado(imagen)
        cv2.imshow("img_filtrada", imagen_filtrada)  # Muestra la imagen filtrada (para debug)
        # Se buscan los contornos en la imagen filtrada
        contornos, _ = cv2.findContours(imagen_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return 0, 0, 0, 0  # Si no se encuentran contornos, retorna ceros

        # Selecciona el contorno con mayor área, asumiendo que es el objeto principal
        contorno = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(contorno)               # Calcula el área del contorno
        perimetro = cv2.arcLength(contorno, True)        # Calcula el perímetro del contorno
        # Calcula la circularidad: 4*pi*área / (perímetro^2)
        circularidad = 4 * math.pi * (area / (perimetro * perimetro)) if perimetro != 0 else 0

        # Obtiene el rectángulo delimitador para calcular la relación de aspecto
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = w / float(h) if h != 0 else 0

        # Calcula la excentricidad usando fitEllipse (si hay suficientes puntos en el contorno)
        excentricidad = 0
        if len(contorno) >= 5:
            (x_elipse, y_elipse), (eje1, eje2), angulo = cv2.fitEllipse(contorno)
            major = max(eje1, eje2)   # Eje mayor
            minor = min(eje1, eje2)   # Eje menor
            if major != 0:
                ratio = (minor / major)**2
                if ratio > 1:
                    ratio = 1         # Asegura que la razón no exceda 1
                excentricidad = math.sqrt(1 - ratio)  # Fórmula de excentricidad
            else:
                excentricidad = 0

        # Calcula los momentos del contorno y obtiene el primer momento de Hu
        momentos = cv2.moments(contorno)
        hu = cv2.HuMoments(momentos)
        hu0 = -1 if hu is None else hu[0][0]

        return circularidad, aspect_ratio, excentricidad, hu0

    def generar_imagen_contorno(self, imagen_cv):
        """
        Genera una imagen con el contorno principal dibujado sobre la imagen original.
        """
        # Aplica el filtrado para obtener la imagen de bordes
        imagen_filtrada = self.filtrado(imagen_cv)
        # Encuentra los contornos en la imagen filtrada
        contornos, _ = cv2.findContours(imagen_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Hace una copia de la imagen original (redimensionada a 400x400)
        contorno_img = self.resize(imagen_cv.copy(), 400, 400)
        if contornos:
            # Selecciona el contorno de mayor área
            contorno_principal = max(contornos, key=cv2.contourArea)
            # Dibuja ese contorno en color verde sobre la imagen
            cv2.drawContours(contorno_img, [contorno_principal], -1, (0, 255, 0), 2)
        return contorno_img

    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        """
        Redimensiona una imagen manteniendo la relación de aspecto.
        :param image: Imagen original.
        :param width: Nuevo ancho (opcional).
        :param height: Nueva altura (opcional).
        :param inter: Método de interpolación (por defecto cv2.INTER_AREA).
        :return: Imagen redimensionada.
        Se prioriza 'height' sobre 'width' si ambos están especificados.
        """
        (original_height, original_width) = image.shape[:2]  # Obtiene dimensiones originales
        if width is None and height is None:
            return image  # Si no se especifica ningún parámetro, retorna la imagen original
        if width is None:
            ratio = height / float(original_height)  # Calcula la proporción según la nueva altura
            width = int(original_width * ratio)        # Calcula el nuevo ancho
        else:
            ratio = width / float(original_width)      # Calcula la proporción según el nuevo ancho
            height = int(original_height * ratio)        # Calcula la nueva altura
        new_size = (width, height)  # Define el nuevo tamaño
        return cv2.resize(image, new_size, interpolation=inter)  # Redimensiona y retorna la imagen

    def filtrado(self, img):
        """
        Aplica un pipeline de procesamiento a la imagen para detectar bordes.
        Este método incluye redimensionamiento, conversión a escala de grises,
        filtrado (mediana), umbral adaptativo, operaciones morfológicas y detección de bordes con Canny.
        """
        # Redimensiona la imagen a 400x400
        img = self.resize(img, width=400, height=400)
        # Convierte la imagen a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("img_gris", gray)  # Muestra la imagen en escala de grises (debug)
        # Aplica un filtro de mediana para reducir el ruido
        gray = cv2.medianBlur(gray, 5)
        cv2.imshow("img_grisBlur", gray)  # Muestra la imagen suavizada
        # Aplica un umbral adaptativo para resaltar el objeto
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5
        )
        cv2.imshow("img_thresh", thresh)  # Muestra la imagen umbralizada
        # Aplica una operación morfológica de cierre para rellenar huecos
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Calcula la mediana de los valores de gris para adaptar los umbrales de Canny
        median_val = np.median(gray)
        lower = int(max(0, (1.0 - 0.33) * median_val))
        upper = int(min(255, (1.0 + 0.33) * median_val))
        # Aplica el detector de bordes Canny
        edges = cv2.Canny(thresh, lower, upper)
        cv2.imshow("img_Canny", edges)  # Muestra la imagen con bordes detectados
        # Aplica nuevamente una operación de cierre para refinar los bordes
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imshow("img_morphClos", edges)  # Muestra la imagen final después de la morfología
        return edges  # Retorna la imagen procesada con los bordes

    def mofologicTransform(self, img, operation, iterations=2):
        """
        Aplica una transformación morfológica a la imagen.
        :param img: Imagen de entrada.
        :param operation: Tipo de operación: "erosion", "dilatacion", "opening" o "closing".
        :param iterations: Número de iteraciones (por defecto 2, aunque se redefinen a 1).
        :return: Imagen transformada.
        """
        sizeKernel = np.ones((4, 3), np.uint8)  # Define el kernel morfológico de tamaño 4x3
        iterations = 1  # Fuerza a usar 1 iteración (puede ajustarse)
        if operation == "erosion":
            operacion = cv2.erode(img, sizeKernel, iterations)
        elif operation == "dilatacion":
            operacion = cv2.dilate(img, sizeKernel, iterations)
        elif operation == "opening":
            operacion = cv2.morphologyEx(img, cv2.MORPH_OPEN, sizeKernel, iterations)
        elif operation == "closing":
            operacion = cv2.morphologyEx(img, cv2.MORPH_CLOSE, sizeKernel, iterations)
        else:
            print("operacion invalida")
            operacion = None
        return operacion
