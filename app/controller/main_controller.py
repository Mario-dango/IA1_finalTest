import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from model.image_model import ImageModel
from model.database_model import DatabaseModel
from model.prediction_model import PredictionModel
from model.astar_model import AStarModel
from model.strips_model import StripsModel
from view.image_view import Ventana2D

class MainController:
    def __init__(self, modelo, vista):
        self.modelo = modelo
        self.vista = vista
        
        self.ultimos_datos = []
        self.ultimas_asignaciones = None
        self.ultimos_centroides_unnorm = None

        self.orden_clases = sorted(["arandelas", "clavos", "tornillos", "tuercas"])
        self.conectar_eventos()

        self.astar_model = AStarModel(self.vista.grid_widget.grid)
        self.modelo["astar"] = self.astar_model
        # Conecta el modelo y la vista de A*
        self.vista.grid_widget.costs = self.astar_model.costs
        self.vista.grid_widget.cost_update_callback = self.update_astar_cost
        self.vista.grid_widget.heuristic_mods = self.astar_model.heuristic_mods
        self.vista.grid_widget.heuristic_update_callback = self.update_astar_heuristic

        self.strips_model = StripsModel()
        self.current_order = ["tornillos", "clavos", "tuercas", "arandelas"]
        self.vista.strips_view.set_current_order(self.current_order)
        default_goal = ["arandelas", "tornillos", "tuercas", "clavos"]
        self.vista.strips_view.set_goal_order(default_goal)
        self.vista.strips_view.button_plan.clicked.connect(self.calcular_plan_strips)

    def conectar_eventos(self):
        self.vista.boton_cargar.clicked.connect(self.cargar_imagen)
        self.vista.boton_procesar.clicked.connect(self.procesar_imagen)
        self.vista.boton_dataset.clicked.connect(self.cargar_dataset)
        self.vista.button_astar.clicked.connect(self.ejecutar_astar)
        self.vista.button_astar_clear.clicked.connect(self.reset_astar_grid)
        
        self.vista.btn_comp1.clicked.connect(self.mostrar_comp1)
        self.vista.btn_comp2.clicked.connect(self.mostrar_comp2)
        self.vista.btn_comp3.clicked.connect(self.mostrar_comp3)

    # --- Lógica para Ventanas 2D ---
    def lanzar_ventana_2d(self, idx_x, idx_y, nombre_x, nombre_y, titulo):
        datos = self.ultimos_datos if self.ultimos_datos else self.modelo["database"].datos_entrenamiento
        
        if not datos:
            self.vista.agregar_a_registro("No hay datos para graficar. Carga el dataset.")
            return

        ventana = Ventana2D(titulo, nombre_x, nombre_y, self.vista)
        ventana.graficar(
            datos=datos,
            idx_x=idx_x, 
            idx_y=idx_y, 
            centroides=self.ultimos_centroides_unnorm,
            asignaciones=self.ultimas_asignaciones,
            nombres_cluster=self.orden_clases
        )
        ventana.exec_()

    def mostrar_comp1(self):
        self.lanzar_ventana_2d(0, 1, "Hu[0]", "Solidez", "Comparación: Solidez vs Hu0")

    def mostrar_comp2(self):
        self.lanzar_ventana_2d(0, 2, "Hu[0]", "Circularidad", "Comparación: Circularidad vs Hu0")

    def mostrar_comp3(self):
        self.lanzar_ventana_2d(1, 2, "Solidez", "Circularidad", "Comparación: Solidez vs Circularidad")
    # -------------------------------

    def cargar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(None, "Seleccionar imagen", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if ruta:
            self.imagen_actual = cv2.imread(ruta)
            if self.imagen_actual is not None:
                self.vista.mostrar_imagen(self.imagen_actual, self.vista.label_imagen)
                self.vista.agregar_a_registro(f"Imagen cargada: {ruta}")
            else:
                self.vista.agregar_a_registro("Error al cargar la imagen.")

    def cargar_dataset(self):
        ruta = QFileDialog.getExistingDirectory(None, "Seleccionar carpeta del dataset")
        if ruta:
            cantidad = self.modelo["database"].cargar_dataset(ruta)
            self.vista.agregar_a_registro(f"Dataset cargado: {cantidad} muestras.")
            
            if cantidad > 0:
                self.ultimos_datos = self.modelo["database"].datos_entrenamiento
                
                self.vista.grafico_knn.plot_puntos(
                    self.ultimos_datos,
                    titulo="Dataset (Etiquetas Reales)"
                )
                
                datos_3d = [fila[:3] for fila in self.ultimos_datos]
                centroides_init = self.calcular_centroides_iniciales(self.orden_clases)
                
                _, asignaciones = self.modelo["prediction"].kmeans_manual(
                    datos_3d, 
                    k=len(self.orden_clases), 
                    max_iter=20,
                    centroides_iniciales=centroides_init
                )
                
                self.ultimas_asignaciones = asignaciones
                self.ultimos_centroides_unnorm = self.modelo["prediction"].get_centroids_unnorm()
                
                self.vista.grafico_kmeans.plot_puntos(
                    self.ultimos_datos,
                    centroides=self.ultimos_centroides_unnorm,
                    asignaciones=self.ultimas_asignaciones,
                    nombres_cluster=self.orden_clases,
                    titulo="K-means (Clusters)"
                )

    def procesar_imagen(self):
        if not hasattr(self, 'imagen_actual') or self.imagen_actual is None:
            self.vista.agregar_a_registro("Primero carga una imagen.")
            return

        h0, solidity, circularity = self.modelo["image"].calcular_caracteristicas(self.imagen_actual)
        self.vista.agregar_a_registro(f"Características -> Hu0: {h0:.4f}, Solidez: {solidity:.4f}, Circ: {circularity:.4f}")

        contorno_img = self.modelo["image"].generar_imagen_contorno(self.imagen_actual)
        self.vista.mostrar_imagen(contorno_img, self.vista.label_contorno)

        # KNN
        knn_pred = self.modelo["prediction"].knn_manual([h0, solidity, circularity])
        
        # K-means
        datos_con_nuevo = self.modelo["database"].datos_entrenamiento.copy()
        datos_con_nuevo.append([h0, solidity, circularity, "Muestra Actual"]) 
        datos_3d = [fila[:3] for fila in datos_con_nuevo]
        
        centroides_init = self.calcular_centroides_iniciales(self.orden_clases)
        
        _, asignaciones = self.modelo["prediction"].kmeans_manual(
            datos_3d, 
            k=len(self.orden_clases), 
            centroides_iniciales=centroides_init
        )
        
        self.ultimos_datos = datos_con_nuevo
        self.ultimas_asignaciones = asignaciones
        self.ultimos_centroides_unnorm = self.modelo["prediction"].get_centroids_unnorm()
        
        cluster_id_nuevo = asignaciones[-1]
        if cluster_id_nuevo < len(self.orden_clases):
            kmeans_pred = self.orden_clases[cluster_id_nuevo]
        else:
            kmeans_pred = f"Cluster {cluster_id_nuevo}"

        # --- ACTUALIZACIÓN DE INTERFAZ Y LOG ---
        self.vista.set_prediccion(f"KNN: {knn_pred} | K-means: {kmeans_pred}")
        self.vista.agregar_a_registro(f"RESULTADO -> KNN: {knn_pred} | K-means: {kmeans_pred}") # Log solicitado
        
        self.vista.grafico_kmeans.plot_puntos(
            datos_con_nuevo,
            centroides=self.ultimos_centroides_unnorm,
            asignaciones=asignaciones,
            nombres_cluster=self.orden_clases,
            titulo="K-means (con predicción)"
        )

    def calcular_centroides_iniciales(self, lista_clases):
        centroides = []
        db = self.modelo["database"]
        if not db.datos_entrenamiento:
            return None

        for clase_nombre in lista_clases:
            filas_clase = [f[:3] for f in db.datos_entrenamiento if f[3] == clase_nombre]
            if filas_clase:
                promedio = np.mean(filas_clase, axis=0)
                promedio_norm = db.normalizar(promedio)
                centroides.append(promedio_norm)
            else:
                centroides.append(np.zeros(3))
        return centroides

    def ejecutar_astar(self):
        grid_widget = self.vista.grid_widget
        start = grid_widget.start
        goal = grid_widget.goal
        if start is None or goal is None:
            self.vista.agregar_a_registro("Define inicio (verde) y fin (rojo).")
            return
        ruta = self.modelo["astar"].astar_search(start, goal)
        if ruta:
            grid_widget.path = ruta
            grid_widget.update()
            self.vista.agregar_a_registro(f"Ruta A*: {len(ruta)} pasos.")
        else:
            self.vista.agregar_a_registro("No hay camino posible.")

    def update_astar_cost(self, row, col, cost):
        self.astar_model.set_cost(row, col, cost)
        self.vista.agregar_a_registro(f"Costo de celda ({row}, {col}) actualizado a {cost}.")
        # No es necesario repintar manualmente si la vista se actualiza sola.

    def update_astar_heuristic(self, row, col, value):
        self.astar_model.set_heuristic_mod(row, col, value)
        self.vista.agregar_a_registro(f"Modificador de heurística de celda ({row}, {col}) actualizado a {value}.")

    def reset_astar_grid(self):
        self.astar_model.reset()
        grid_widget = self.vista.grid_widget
        grid_widget.start = None
        grid_widget.goal = None
        grid_widget.path = []
        grid_widget.update()
        self.vista.agregar_a_registro("Grilla de A* restablecida.")

    def calcular_plan_strips(self):
        goal_order = self.vista.strips_view.get_goal_order()
        plan = self.strips_model.plan_reordering(self.current_order, goal_order)
        self.vista.strips_view.display_plan(plan)
        d_str = self.strips_model.get_domain_str()
        p_str = self.strips_model.get_problem_str(self.current_order, goal_order)
        self.vista.text_strips_domain.setText(d_str)
        
        self.vista.text_strips_problem.setText(p_str)