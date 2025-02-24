from PyQt5.QtWidgets import QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # Necesario para 3D
import numpy as np

class Grafico3DCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Inicializa un widget para mostrar gráficos 3D.
        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)
        # Conectar el evento de scroll
        self.mpl_connect("scroll_event", self.on_scroll)

    def plot_puntos(self, datos, centroides=None, asignaciones=None, titulo="Gráfico 3D"):
        """
        Grafica puntos en 3D y opcionalmente los centroides.
        """
        self.axes.clear()
        self.axes.set_title(titulo)
        self.axes.set_xlabel("Circularidad")
        self.axes.set_ylabel("Aspect Ratio")
        self.axes.set_zlabel("Hu[0]")

        if asignaciones is not None:
            # Graficar puntos por cluster
            asignaciones = np.array(asignaciones)
            for cluster_id in np.unique(asignaciones):
                cluster_puntos = np.array(
                    [d[:3] for i, d in enumerate(datos) if asignaciones[i] == cluster_id]
                )
                if len(cluster_puntos) > 0:
                    self.axes.scatter(
                        cluster_puntos[:, 0], cluster_puntos[:, 1], cluster_puntos[:, 2],
                        label=f"Cluster {cluster_id}"
                    )
            # Graficar centroides
            if centroides is not None:
                self.axes.scatter(
                    centroides[:, 0], centroides[:, 1], centroides[:, 2],
                    c='black', marker='X', s=100, label="Centroides"
                )
        else:
            # Graficar puntos por etiqueta
            etiquetas = set([d[3] for d in datos])
            colores = ["red", "green", "blue", "orange", "magenta", "cyan"]
            col_idx = 0
            for etiq in etiquetas:
                subset = np.array([d[:3] for d in datos if d[3] == etiq])
                self.axes.scatter(
                    subset[:, 0], subset[:, 1], subset[:, 2],
                    color=colores[col_idx % len(colores)],
                    label=str(etiq)
                )
                col_idx += 1

        self.axes.legend(
                loc='upper left', 
                bbox_to_anchor=(1.05, 1.0), 
                borderaxespad=0.
            )

        self.draw()


    def on_scroll(self, event):
        """Callback para manejar el zoom con la rueda del mouse."""
        base_scale = 1.1  # Factor de zoom
        # Obtener el rango actual de los ejes
        x_left, x_right = self.axes.get_xlim3d()
        y_left, y_right = self.axes.get_ylim3d()
        z_left, z_right = self.axes.get_zlim3d()
        
        x_range = x_right - x_left
        y_range = y_right - y_left
        z_range = z_right - z_left

        # Si el scroll es hacia arriba, se hace zoom in; si es hacia abajo, zoom out
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        # Calcular nuevos límites centrados en el punto medio
        x_mid = (x_left + x_right) * 0.5
        y_mid = (y_left + y_right) * 0.5
        z_mid = (z_left + z_right) * 0.5

        new_x_range = x_range * scale_factor
        new_y_range = y_range * scale_factor
        new_z_range = z_range * scale_factor

        self.axes.set_xlim3d([x_mid - new_x_range/2, x_mid + new_x_range/2])
        self.axes.set_ylim3d([y_mid - new_y_range/2, y_mid + new_y_range/2])
        self.axes.set_zlim3d([z_mid - new_z_range/2, z_mid + new_z_range/2])
        self.draw()
