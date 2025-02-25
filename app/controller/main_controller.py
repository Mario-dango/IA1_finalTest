import cv2                                 # Importa OpenCV para procesamiento de imágenes
import numpy as np                         # Importa NumPy para cálculos numéricos y manejo de arrays
from PyQt5.QtWidgets import QFileDialog      # Importa QFileDialog para diálogos de selección de archivos/carpetas
from model.image_model import ImageModel      # Importa el modelo de imágenes (procesamiento y extracción de características)
from model.database_model import DatabaseModel  # Importa el modelo de base de datos (carga de dataset y normalización)
from model.prediction_model import PredictionModel  # Importa el modelo de predicción (KNN, K-means)
from model.astar_model import AStarModel      # Importa el modelo A* para búsqueda de rutas
from model.strips_model import StripsModel    # Importa el modelo STRIPS para planificación

class MainController:
    def __init__(self, modelo, vista):
        """
        Inicializa el controlador principal y conecta la lógica de la aplicación.
        
        Parámetros:
         - modelo: Diccionario que contiene instancias de los modelos ("image", "database", "prediction", etc.).
         - vista: Instancia de la ventana principal (MainWindow) que contiene la interfaz.
        """
        self.modelo = modelo   # Guarda el diccionario de modelos
        self.vista = vista     # Guarda la referencia a la vista (ventana principal)
        self.conectar_eventos()  # Conecta los eventos de la vista a los métodos del controlador

        # Configuración de A*:
        # Se crea una instancia de AStarModel utilizando la grilla definida en el widget interactivo de A*
        self.astar_model = AStarModel(self.vista.grid_widget.grid)
        self.modelo["astar"] = self.astar_model  # Se añade la instancia de A* al diccionario de modelos

        # Configuración de STRIPS:
        # Se crea una instancia de StripsModel para planificar el reordenamiento
        self.strips_model = StripsModel()
        # Se simula el estado actual (detectado mediante visión) de la pila de cajas
        self.current_order = ["tornillos", "clavos", "tuercas", "arandelas"]
        # Se actualiza la vista STRIPS:
        # Se establece el orden actual en el widget STRIPS (lista de solo lectura)
        self.vista.strips_view.set_current_order(self.current_order)
        # Se define un orden objetivo por defecto (editable por el usuario)
        default_goal = ["arandelas", "tornillos", "tuercas", "clavos"]
        self.vista.strips_view.set_goal_order(default_goal)
        # Conecta el botón de "Calcular Plan STRIPS" del widget STRIPS con el método que calcula el plan
        self.vista.strips_view.button_plan.clicked.connect(self.calcular_plan_strips)

    def conectar_eventos(self):
        """
        Conecta los eventos (clics de botones) de la vista a los métodos correspondientes en el controlador.
        """
        self.vista.boton_cargar.clicked.connect(self.cargar_imagen)      # Conecta el botón para cargar imagen
        self.vista.boton_procesar.clicked.connect(self.procesar_imagen)   # Conecta el botón para procesar imagen
        self.vista.boton_dataset.clicked.connect(self.cargar_dataset)     # Conecta el botón para cargar el dataset
        self.vista.button_astar.clicked.connect(self.ejecutar_astar)      # Conecta el botón para ejecutar A*

    def cargar_imagen(self):
        """
        Permite al usuario seleccionar y cargar una imagen desde el sistema de archivos.
        La imagen se lee con OpenCV y se muestra en la vista en el área correspondiente.
        """
        # Abre un diálogo para seleccionar una imagen con las extensiones indicadas
        ruta, _ = QFileDialog.getOpenFileName(None, "Seleccionar imagen", "", "Images (*.png *.jpg *.bmp)")
        if ruta:
            self.imagen_actual = cv2.imread(ruta)  # Lee la imagen en formato BGR
            if self.imagen_actual is not None:
                # Muestra la imagen original en el QLabel correspondiente de la pestaña "Detección"
                self.vista.mostrar_imagen(self.imagen_actual, self.vista.label_imagen)
                self.vista.agregar_a_registro(f"Imagen cargada: {ruta}")  # Registra el evento
            else:
                self.vista.agregar_a_registro("Error al cargar la imagen.")
        else:
            self.vista.agregar_a_registro("No se seleccionó ninguna imagen.")

    def cargar_dataset(self):
        """
        Permite al usuario seleccionar una carpeta que contenga el dataset de imágenes.
        Se procesan las imágenes para extraer sus características, se actualiza el dataset
        y se actualizan los gráficos 3D de KNN y K-means.
        """
        # Abre un diálogo para seleccionar la carpeta del dataset
        ruta = QFileDialog.getExistingDirectory(None, "Seleccionar carpeta del dataset")
        if ruta:
            # Carga el dataset y obtiene el número de imágenes cargadas
            cantidad = self.modelo["database"].cargar_dataset(ruta)
            self.vista.agregar_a_registro(f"Dataset cargado desde: {ruta} con {cantidad} muestras.")
            if self.modelo["database"].datos_entrenamiento:
                # Actualiza el gráfico KNN sin asignaciones (simple dispersión)
                self.vista.grafico_knn.plot_puntos(
                    self.modelo["database"].datos_entrenamiento,
                    titulo="Dataset (KNN)"
                )
                # Prepara los datos para K-means: toma las 3 primeras características de cada muestra
                datos_3d = [fila[:3] for fila in self.modelo["database"].datos_entrenamiento]
                # Ejecuta K-means de forma manual
                _, asignaciones = self.modelo["prediction"].kmeans_manual(datos_3d, k=4, max_iter=10)
                # Obtiene los centroides desnormalizados para graficarlos en el espacio original
                centroides_unnorm = self.modelo["prediction"].get_centroids_unnorm()
                # Actualiza el gráfico K-means mostrando clusters y centroides
                self.vista.grafico_kmeans.plot_puntos(
                    self.modelo["database"].datos_entrenamiento,
                    centroides=centroides_unnorm,
                    asignaciones=asignaciones,
                    titulo="Dataset (K-means)"
                )
        else:
            self.vista.agregar_a_registro("No se seleccionó carpeta para el dataset.")

    def procesar_imagen(self):
        """
        Procesa la imagen cargada:
          - Extrae las características (circularidad, aspect ratio, etc.) utilizando ImageModel.
          - Genera y muestra la imagen con el contorno dibujado.
          - Realiza la clasificación con el algoritmo k-NN.
          - Actualiza la predicción en la vista.
        """
        # Verifica que se haya cargado una imagen previamente
        if not hasattr(self, 'imagen_actual') or self.imagen_actual is None:
            self.vista.agregar_a_registro("No hay imagen para procesar. Carga una primero.")
            return

        # Extrae las características de la imagen
        circ, asp, exc, hu0 = self.modelo["image"].calcular_caracteristicas(self.imagen_actual)
        # Registra las características extraídas en el área de registro
        self.vista.agregar_a_registro(
            f"Características extraídas -> Circularidad: {circ:.3f}, Aspect Ratio: {asp:.3f}, "
            f"Excentricidad: {exc:.3f}, Hu[0]: {hu0:.6f}"
        )
        # Genera la imagen con el contorno dibujado y la muestra en la vista (pestaña "Detección")
        contorno_img = self.modelo["image"].generar_imagen_contorno(self.imagen_actual)
        self.vista.mostrar_imagen(contorno_img, self.vista.label_contorno)
        # Clasifica la imagen usando k-NN (se utiliza un vector con 3 características)
        etiqueta_knn = self.modelo["prediction"].knn_manual([circ, asp, hu0])
        self.vista.agregar_a_registro(f"KNN dice que es: {etiqueta_knn}")
        # Actualiza la etiqueta de predicción en la vista
        self.vista.set_prediccion(etiqueta_knn)

    def ejecutar_astar(self):
        """
        Ejecuta el algoritmo A* para encontrar la ruta más corta en la grilla interactiva.
        Utiliza la posición de inicio y destino seleccionadas por el usuario.
        """
        grid_widget = self.vista.grid_widget  # Obtiene el widget de la grilla A*
        start = grid_widget.start             # Obtiene la posición de inicio (origen)
        goal = grid_widget.goal               # Obtiene la posición de destino (meta)
        if start is None or goal is None:
            # Si no se han seleccionado ambos puntos, registra un mensaje de error
            self.vista.agregar_a_registro("Seleccione la posición de origen y destino en la grilla.")
            return

        self.vista.agregar_a_registro(f"Ejecutando A* desde {start} hasta {goal}...")
        # Ejecuta el algoritmo A* usando el modelo configurado
        ruta = self.modelo["astar"].astar_search(start, goal)
        if ruta is None:
            self.vista.agregar_a_registro("No se encontró ruta con A*.")
        else:
            self.vista.agregar_a_registro(f"Ruta encontrada: {ruta}")
            grid_widget.path = ruta   # Asigna la ruta calculada al widget de la grilla
            grid_widget.update()       # Solicita redibujar el widget para mostrar la ruta

    def calcular_plan_strips(self):
        """
        Calcula el plan STRIPS basado en el estado actual de la pila de cajas (current_order)
        y el orden objetivo definido por el usuario en el widget STRIPS.
        Actualiza el widget STRIPS y las áreas de texto para mostrar la definición del dominio
        y el problema en lenguaje STRIPS.
        """
        # Obtiene el orden objetivo (definido por el usuario en el widget STRIPS)
        goal_order = self.vista.strips_view.get_goal_order()
        # Calcula la secuencia de acciones para reordenar las cajas usando el modelo STRIPS
        plan = self.strips_model.plan_reordering(self.current_order, goal_order)
        # Muestra el plan en el widget STRIPS (área de texto interna del widget)
        self.vista.strips_view.display_plan(plan)
        # Obtiene la definición del dominio en lenguaje STRIPS
        domain_str = self.strips_model.get_domain_str()
        # Obtiene la definición del problema STRIPS basado en el estado actual y el objetivo
        problem_str = self.strips_model.get_problem_str(self.current_order, goal_order)
        # Actualiza las áreas de texto en la pestaña STRIPS con el dominio y problema generados
        self.vista.text_strips_domain.setText(domain_str)
        self.vista.text_strips_problem.setText(problem_str)
        self.vista.agregar_a_registro("Plan STRIPS calculado y mostrado en la pestaña STRIPS.")
