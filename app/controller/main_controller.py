from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np

class MainController:
    def __init__(self, modelo, vista):
        """
        Inicializa el controlador principal.
        """
        self.modelo = modelo
        self.vista = vista
        self.conectar_eventos()

    def conectar_eventos(self):
        """
        Conecta los eventos de la vista con los métodos del controlador.
        """
        self.vista.boton_cargar.clicked.connect(self.cargar_imagen)
        self.vista.boton_procesar.clicked.connect(self.procesar_imagen)
        self.vista.boton_dataset.clicked.connect(self.cargar_dataset)

    def cargar_imagen(self):
        """
        Carga una imagen desde un archivo.
        """
        ruta, _ = QFileDialog.getOpenFileName(None, "Seleccionar imagen", "", "Images (*.png *.jpg *.bmp)")
        if ruta:
            self.imagen_actual = cv2.imread(ruta)
            if self.imagen_actual is not None:
                self.vista.mostrar_imagen(self.imagen_actual, self.vista.label_imagen)
                self.vista.agregar_a_registro(f"Imagen cargada: {ruta}")
            else:
                self.vista.agregar_a_registro("Error al cargar la imagen.")
        else:
            self.vista.agregar_a_registro("No se seleccionó ninguna imagen.")

    def cargar_dataset(self):
        """
        Carga un dataset desde una carpeta.
        """
        ruta = QFileDialog.getExistingDirectory(None, "Seleccionar carpeta del dataset")
        if ruta:
            cantidad = self.modelo["database"].cargar_dataset(ruta)
            self.vista.agregar_a_registro(f"Dataset cargado desde: {ruta} con {cantidad} muestras.")
            if self.modelo["database"].datos_entrenamiento:
                # Graficar KNN (sin asignaciones)
                self.vista.grafico_knn.plot_puntos(
                    self.modelo["database"].datos_entrenamiento,  # en escala original
                    titulo="Dataset (KNN)"
                )
                # Hacer K-means
                datos_3d = [fila[:3] for fila in self.modelo["database"].datos_entrenamiento]
                _, asignaciones = self.modelo["prediction"].kmeans_manual(datos_3d, k=4, max_iter=10)
                # Des-normalizar los centroides para graficar
                centroides_unnorm = self.modelo["prediction"].get_centroids_unnorm()
                self.vista.grafico_kmeans.plot_puntos(
                    self.modelo["database"].datos_entrenamiento,  # en escala original
                    centroides=centroides_unnorm,  # en escala original
                    asignaciones=asignaciones,
                    titulo="Dataset (K-means)"
                )
        else:
            self.vista.agregar_a_registro("No se seleccionó carpeta para el dataset.")

    def procesar_imagen(self):
        """
        Procesa la imagen actual y realiza la predicción.
        """
        if not hasattr(self, 'imagen_actual') or self.imagen_actual is None:
            self.vista.agregar_a_registro("No hay imagen para procesar. Carga una primero.")
            return

        # Calcular características de la imagen
        circ, asp, exc, hu0 = self.modelo["image"].calcular_caracteristicas(self.imagen_actual)
        self.vista.agregar_a_registro(
            f"Características extraídas -> Circularidad: {circ:.3f}, "
            f"Aspect Ratio: {asp:.3f}, Excentricidad: {exc:.3f}, Hu[0]: {hu0:.6f}"
        )

        # Mostrar imagen de contorno
        contorno_img = self.modelo["image"].generar_imagen_contorno(self.imagen_actual)
        self.vista.mostrar_imagen(contorno_img, self.vista.label_contorno)

        # Clasificación KNN
        etiqueta_knn = self.modelo["prediction"].knn_manual([circ, asp, hu0])
        self.vista.agregar_a_registro(f"KNN dice que es: {etiqueta_knn}")

        # Agregar nuevo punto al dataset (temporal) para K-means
        datos_con_nuevo = self.modelo["database"].datos_entrenamiento.copy()
        datos_con_nuevo.append([circ, asp, hu0, "Desconocido"])
        datos_3d = [fila[:3] for fila in datos_con_nuevo]

        # K-means en el espacio normalizado
        _, asignaciones = self.modelo["prediction"].kmeans_manual(datos_3d, k=4, max_iter=20)
        cluster_nuevo = asignaciones[-1]

        # Mapear cada cluster al nombre mayoritario según el dataset (excluyendo "Desconocido")
        cluster_mapping = {}
        clusters = np.unique(asignaciones[:-1])  # solo de las muestras de entrenamiento
        for cl in clusters:
            etiquetas = [datos_con_nuevo[i][3] for i in range(len(asignaciones)-1) if asignaciones[i]==cl]
            if etiquetas:
                # Votación mayoritaria
                cluster_mapping[cl] = max(set(etiquetas), key=etiquetas.count)
        # Nombre del cluster al que fue asignado
        cluster_nombre = cluster_mapping.get(cluster_nuevo, "Desconocido")

        # Des-normalizar centroides para graficarlos en el espacio original
        centroides_unnorm = self.modelo["prediction"].get_centroids_unnorm()
        self.vista.grafico_kmeans.plot_puntos(
            datos_con_nuevo,  # en escala original
            centroides=centroides_unnorm,  # en escala original
            asignaciones=asignaciones,
            titulo="K-means (con nuevo punto)"
        )

        # Mostrar predicción
        mensaje = f"Es {etiqueta_knn} (K-means sugiere '{cluster_nombre}')."
        self.vista.set_prediccion(mensaje)