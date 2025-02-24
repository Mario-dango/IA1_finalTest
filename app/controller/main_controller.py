import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from model.image_model import ImageModel
from model.database_model import DatabaseModel
from model.prediction_model import PredictionModel
from model.astar_model import AStarModel
from model.strips_model import StripsModel

class MainController:
    def __init__(self, modelo, vista):
        """
        Inicializa el controlador, conecta los eventos y configura los modelos.
        Se integra la lógica de Detección, Dataset, A* y STRIPS.
        """
        self.modelo = modelo
        self.vista = vista
        self.conectar_eventos()

        # Configurar A* usando la grilla interactiva de la vista
        self.astar_model = AStarModel(self.vista.grid_widget.grid)
        self.modelo["astar"] = self.astar_model

        # Configurar STRIPS
        self.strips_model = StripsModel()
        # Simular el estado actual (detectado por visión)
        self.current_order = ["tornillos", "clavos", "tuercas", "arandelas"]
        # Actualizar la vista STRIPS con el estado actual y un orden objetivo por defecto
        self.vista.strips_view.set_current_order(self.current_order)
        default_goal = ["arandelas", "tornillos", "tuercas", "clavos"]
        self.vista.strips_view.set_goal_order(default_goal)
        # Conectar el botón de STRIPS
        self.vista.strips_view.button_plan.clicked.connect(self.calcular_plan_strips)

    def conectar_eventos(self):
        self.vista.boton_cargar.clicked.connect(self.cargar_imagen)
        self.vista.boton_procesar.clicked.connect(self.procesar_imagen)
        self.vista.boton_dataset.clicked.connect(self.cargar_dataset)
        self.vista.button_astar.clicked.connect(self.ejecutar_astar)

    def cargar_imagen(self):
        """Permite al usuario cargar una imagen."""
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
        """Permite al usuario seleccionar y cargar el dataset y actualiza los gráficos 3D."""
        ruta = QFileDialog.getExistingDirectory(None, "Seleccionar carpeta del dataset")
        if ruta:
            cantidad = self.modelo["database"].cargar_dataset(ruta)
            self.vista.agregar_a_registro(f"Dataset cargado desde: {ruta} con {cantidad} muestras.")
            if self.modelo["database"].datos_entrenamiento:
                self.vista.grafico_knn.plot_puntos(
                    self.modelo["database"].datos_entrenamiento,
                    titulo="Dataset (KNN)"
                )
                datos_3d = [fila[:3] for fila in self.modelo["database"].datos_entrenamiento]
                _, asignaciones = self.modelo["prediction"].kmeans_manual(datos_3d, k=4, max_iter=10)
                centroides_unnorm = self.modelo["prediction"].get_centroids_unnorm()
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
        Procesa la imagen:
          - Extrae características (usando ImageModel).
          - Genera y muestra el contorno.
          - Realiza clasificación (por ejemplo, con KNN).
        """
        if not hasattr(self, 'imagen_actual') or self.imagen_actual is None:
            self.vista.agregar_a_registro("No hay imagen para procesar. Carga una primero.")
            return

        circ, asp, exc, hu0 = self.modelo["image"].calcular_caracteristicas(self.imagen_actual)
        self.vista.agregar_a_registro(
            f"Características extraídas -> Circularidad: {circ:.3f}, Aspect Ratio: {asp:.3f}, "
            f"Excentricidad: {exc:.3f}, Hu[0]: {hu0:.6f}"
        )
        contorno_img = self.modelo["image"].generar_imagen_contorno(self.imagen_actual)
        self.vista.mostrar_imagen(contorno_img, self.vista.label_contorno)
        etiqueta_knn = self.modelo["prediction"].knn_manual([circ, asp, hu0])
        self.vista.agregar_a_registro(f"KNN dice que es: {etiqueta_knn}")
        self.vista.set_prediccion(etiqueta_knn)

    def ejecutar_astar(self):
        """
        Ejecuta A* usando las posiciones seleccionadas en la grilla interactiva.
        """
        grid_widget = self.vista.grid_widget
        start = grid_widget.start
        goal = grid_widget.goal
        if start is None or goal is None:
            self.vista.agregar_a_registro("Seleccione la posición de origen y destino en la grilla.")
            return

        self.vista.agregar_a_registro(f"Ejecutando A* desde {start} hasta {goal}...")
        ruta = self.modelo["astar"].astar_search(start, goal)
        if ruta is None:
            self.vista.agregar_a_registro("No se encontró ruta con A*.")
        else:
            self.vista.agregar_a_registro(f"Ruta encontrada: {ruta}")
            grid_widget.path = ruta
            grid_widget.update()

    def calcular_plan_strips(self):
        """
        Calcula el plan STRIPS usando el orden actual y el orden objetivo definido por el usuario.
        Actualiza el widget STRIPS, y las áreas de texto del dominio y del problema.
        """
        goal_order = self.vista.strips_view.get_goal_order()
        plan = self.strips_model.plan_reordering(self.current_order, goal_order)
        self.vista.strips_view.display_plan(plan)
        # Actualizar las áreas de texto de dominio y problema STRIPS
        domain_str = self.strips_model.get_domain_str()
        problem_str = self.strips_model.get_problem_str(self.current_order, goal_order)
        self.vista.text_strips_domain.setText(domain_str)
        self.vista.text_strips_problem.setText(problem_str)
        self.vista.agregar_a_registro("Plan STRIPS calculado y mostrado en la pestaña STRIPS.")
