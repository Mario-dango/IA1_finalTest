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