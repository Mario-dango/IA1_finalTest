from PyQt5.QtWidgets import QLabel, QDialog, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- GRÁFICO 3D PRINCIPAL (Sin cambios mayores, solo mantenemos la clase) ---
class Grafico3DCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)
        self.mpl_connect("scroll_event", self.on_scroll)

    def plot_puntos(self, datos, centroides=None, asignaciones=None, nombres_cluster=None, titulo="Gráfico 3D"):
        self.axes.clear()
        self.axes.set_title(titulo)
        self.axes.set_xlabel("Hu[0] (Log)")
        self.axes.set_ylabel("Solidez")
        self.axes.set_zlabel("Circularidad")

        puntos = np.array([d[:3] for d in datos], dtype=float)
        colores_base = ['r', 'g', 'b', 'y', 'm', 'c']

        if asignaciones is not None:
            asignaciones = np.array(asignaciones)
            unique_clusters = np.unique(asignaciones)
            
            for i, cluster_id in enumerate(unique_clusters):
                mask = (asignaciones == cluster_id)
                cluster_puntos = puntos[mask]
                
                if nombres_cluster and cluster_id < len(nombres_cluster):
                    nombre_leyenda = f"{nombres_cluster[cluster_id]}"
                else:
                    nombre_leyenda = f"Cluster {cluster_id}"
                
                color = colores_base[cluster_id % len(colores_base)]
                
                if len(cluster_puntos) > 0:
                    self.axes.scatter(
                        cluster_puntos[:, 0], cluster_puntos[:, 1], cluster_puntos[:, 2],
                        c=color, label=nombre_leyenda, alpha=0.6, edgecolors='w'
                    )

            if centroides is not None:
                centroides = np.array(centroides)
                self.axes.scatter(
                    centroides[:, 0], centroides[:, 1], centroides[:, 2],
                    c='black', marker='X', s=50, label="Centroides", depthshade=False, alpha=0.8
                )

        else:
            etiquetas_reales = [d[3] for d in datos]
            etiquetas_unicas = sorted(list(set(etiquetas_reales)))
            
            for i, etiqueta in enumerate(etiquetas_unicas):
                indices = [idx for idx, val in enumerate(etiquetas_reales) if val == etiqueta]
                subset = puntos[indices]
                color = colores_base[i % len(colores_base)]
                self.axes.scatter(
                    subset[:, 0], subset[:, 1], subset[:, 2],
                    c=color, label=str(etiqueta), alpha=0.6
                )

        self.axes.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        self.draw()

    def on_scroll(self, event):
        base_scale = 1.1
        x_left, x_right = self.axes.get_xlim3d()
        y_left, y_right = self.axes.get_ylim3d()
        z_left, z_right = self.axes.get_zlim3d()
        if event.button == 'up': scale_factor = 1 / base_scale
        elif event.button == 'down': scale_factor = base_scale
        else: scale_factor = 1
        x_mid = (x_left + x_right) * 0.5
        y_mid = (y_left + y_right) * 0.5
        z_mid = (z_left + z_right) * 0.5
        new_rx = (x_right - x_left) * scale_factor
        new_ry = (y_right - y_left) * scale_factor
        new_rz = (z_right - z_left) * scale_factor
        self.axes.set_xlim3d([x_mid - new_rx/2, x_mid + new_rx/2])
        self.axes.set_ylim3d([y_mid - new_ry/2, y_mid + new_ry/2])
        self.axes.set_zlim3d([z_mid - new_rz/2, z_mid + new_rz/2])
        self.draw()


# --- NUEVA CLASE PARA VENTANAS EMERGENTES 2D MEJORADA ---
class Ventana2D(QDialog):
    def __init__(self, titulo, label_x, label_y, parent=None):
        super().__init__(parent)
        self.setWindowTitle(titulo)
        self.resize(1000, 500) # Más ancho para que quepan dos gráficos
        self.label_x = label_x
        self.label_y = label_y
        
        # Layout principal
        layout = QVBoxLayout(self)
        
        # Figura Matplotlib con 2 subplots (1 fila, 2 columnas)
        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Barra de herramientas para Zoom y Pan
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Crear ejes
        self.ax1 = self.figure.add_subplot(121) # Izquierda: Datos/KNN
        self.ax2 = self.figure.add_subplot(122) # Derecha: K-means
        
        layout.addWidget(self.toolbar) # Agregar barra arriba
        layout.addWidget(self.canvas)

    def graficar(self, datos, idx_x, idx_y, centroides=None, asignaciones=None, nombres_cluster=None):
        """
        :param idx_x: Índice de la columna para el eje X
        :param idx_y: Índice de la columna para el eje Y
        """
        puntos = np.array([d[:3] for d in datos], dtype=float)
        colores_base = ['r', 'g', 'b', 'y', 'm', 'c']
        
        # --- GRÁFICO 1: REALIDAD / KNN (Izquierda) ---
        self.ax1.clear()
        self.ax1.set_title("Datos / Etiquetas Reales (KNN)")
        self.ax1.set_xlabel(self.label_x)
        self.ax1.set_ylabel(self.label_y)
        self.ax1.grid(True, linestyle='--', alpha=0.5)
        
        etiquetas_reales = [d[3] for d in datos]
        etiquetas_unicas = sorted(list(set(etiquetas_reales)))
        
        for i, etiqueta in enumerate(etiquetas_unicas):
            indices = [idx for idx, val in enumerate(etiquetas_reales) if val == etiqueta]
            subset = puntos[indices]
            # Usar colores consistentes si es posible
            color = colores_base[i % len(colores_base)]
            self.ax1.scatter(
                subset[:, idx_x], subset[:, idx_y],
                c=color, label=str(etiqueta), alpha=0.6, edgecolors='k'
            )
        self.ax1.legend()

        # --- GRÁFICO 2: CLUSTERS / K-MEANS (Derecha) ---
        self.ax2.clear()
        self.ax2.set_title("Clusters Asignados (K-means)")
        self.ax2.set_xlabel(self.label_x)
        self.ax2.set_ylabel(self.label_y)
        self.ax2.grid(True, linestyle='--', alpha=0.5)

        if asignaciones is not None:
            asignaciones = np.array(asignaciones)
            unique_clusters = np.unique(asignaciones)
            
            for cluster_id in unique_clusters:
                mask = (asignaciones == cluster_id)
                cluster_puntos = puntos[mask]
                
                if nombres_cluster and cluster_id < len(nombres_cluster):
                    leyenda = f"{nombres_cluster[cluster_id]}"
                else:
                    leyenda = f"Cluster {cluster_id}"
                
                color = colores_base[cluster_id % len(colores_base)]
                
                if len(cluster_puntos) > 0:
                    self.ax2.scatter(
                        cluster_puntos[:, idx_x], cluster_puntos[:, idx_y],
                        c=color, label=leyenda, alpha=0.6, edgecolors='w'
                    )
            
            # Dibujar Centroides en el gráfico de K-means
            if centroides is not None:
                centroides = np.array(centroides)
                self.ax2.scatter(
                    centroides[:, idx_x], centroides[:, idx_y],
                    c='black', marker='X', s=100, label="Centroides", zorder=5
                )
            self.ax2.legend()
        else:
            self.ax2.text(0.5, 0.5, "No hay datos de K-means procesados", 
                          ha='center', va='center', transform=self.ax2.transAxes)

        self.figure.tight_layout() # Ajustar márgenes
        self.canvas.draw()