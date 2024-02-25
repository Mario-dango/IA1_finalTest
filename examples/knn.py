import cv2
import numpy as np
import matplotlib.pyplot as plt

def calcular_momentos_hu(imagen):
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbralización para segmentar la imagen
    _, umbral = cv2.threshold(gris, 128, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tomar el contorno con el área más grande como región de interés (ROI)
    mayor_contorno = max(contornos, key=cv2.contourArea)
    
    # Calcular momentos de Hu
    momentos_hu = cv2.HuMoments(cv2.moments(mayor_contorno)).flatten()
    
    return momentos_hu

def knn_algoritmo(datos_entrenamiento, etiquetas_entrenamiento, muestra_nueva, k=3):
    # Calcular las distancias euclidianas entre la muestra nueva y los datos de entrenamiento
    distancias = np.linalg.norm(datos_entrenamiento - muestra_nueva, axis=1)
    
    # Obtener los índices de las k muestras más cercanas
    indices_k_cercanos = np.argsort(distancias)[:k]
    
    # Obtener las etiquetas correspondientes a las k muestras más cercanas
    etiquetas_k_cercanas = etiquetas_entrenamiento[indices_k_cercanos]
    
    # Determinar la etiqueta más común entre las k muestras cercanas (votación)
    etiqueta_predicha = np.argmax(np.bincount(etiquetas_k_cercanas))
    
    return etiqueta_predicha


# Cargar imágenes de entrenamiento
arandela1 = cv2.imread('../resources/dataset/internet/arandela02.jpg')
arandela2 = cv2.imread('../resources/dataset/internet/arandela01.jpg')
clavo1 = cv2.imread('../resources/dataset/internet/clavo02.jpg')
clavo2 = cv2.imread('../resources/dataset/internet/clavo05.jpg')

# Calcular momentos de Hu para imágenes de entrenamiento
momentos_hu_arandela1 = calcular_momentos_hu(arandela1)
momentos_hu_arandela2 = calcular_momentos_hu(arandela2)
momentos_hu_clavo1 = calcular_momentos_hu(clavo1)
momentos_hu_clavo2 = calcular_momentos_hu(clavo2)

# Datos de entrenamiento y etiquetas
datos_entrenamiento = np.array([momentos_hu_arandela1, momentos_hu_arandela2, momentos_hu_clavo1, momentos_hu_clavo2])
etiquetas_entrenamiento = np.array([0, 0, 1, 1])

# Visualización de momentos de Hu (usando solo los dos primeros momentos)
plt.scatter(datos_entrenamiento[:, 0], datos_entrenamiento[:, 1], c=etiquetas_entrenamiento, cmap='viridis')
plt.title('Visualización de Momentos de Hu (2D)')
plt.xlabel('Momento de Hu 1')
plt.ylabel('Momento de Hu 2')
plt.show()




# Cargar la imagen
# imagen = cv2.imread('../resources/dataset/internet/clavo05.jpg')
imagen = cv2.imread('../resources/dataset/internet/arandela05.jpg')

# Calcular momentos de Hu para la imagen
momentos_hu_imagen = calcular_momentos_hu(imagen)

# Datos de entrenamiento (momentos de Hu) y etiquetas (0 para arandela, 1 para clavo)
datos_entrenamiento = np.array([
    calcular_momentos_hu(cv2.imread("../resources/dataset/internet/arandela02.jpg")),
    calcular_momentos_hu(cv2.imread('../resources/dataset/internet/arandela06.jpg')),
    calcular_momentos_hu(cv2.imread('../resources/dataset/internet/clavo05.jpg')),
    calcular_momentos_hu(cv2.imread('../resources/dataset/internet/clavo02.jpg')),
])
etiquetas_entrenamiento = np.array([0, 0, 1, 1])

# Determinar la etiqueta predicha para la imagen usando k-NN
etiqueta_predicha = knn_algoritmo(datos_entrenamiento, etiquetas_entrenamiento, momentos_hu_imagen)

# Mostrar la imagen con la etiqueta predicha
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title(f'Etiqueta Predicha: {"Arandela" if etiqueta_predicha == 0 else "Clavo"}')
plt.show()
