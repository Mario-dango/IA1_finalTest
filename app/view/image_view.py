from PyQt5.QtWidgets import QLabel                 # Importa QLabel para mostrar imágenes o texto
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Importa la clase base para integrar matplotlib en PyQt
from matplotlib.figure import Figure               # Importa la clase Figure para crear figuras de matplotlib
from mpl_toolkits.mplot3d import Axes3D            # Importa el toolkit 3D para habilitar gráficos 3D (necesario, aunque no se use directamente)
import numpy as np                                 # Importa NumPy para operaciones con arrays

class Grafico3DCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Inicializa un widget para mostrar gráficos 3D.
        :param parent: Widget padre (opcional)
        :param width: Ancho de la figura
        :param height: Alto de la figura
        :param dpi: Resolución de la figura
        """
        # Crea una figura de matplotlib con el tamaño y dpi especificados
        fig = Figure(figsize=(width, height), dpi=dpi)
        # Agrega un subplot 3D a la figura
        self.axes = fig.add_subplot(111, projection='3d')
        # Inicializa el canvas con la figura creada
        super().__init__(fig)
        self.setParent(parent)
        # Conecta el evento de scroll para permitir zoom en el gráfico 3D
        self.mpl_connect("scroll_event", self.on_scroll)

    def plot_puntos(self, datos, centroides=None, asignaciones=None, titulo="Gráfico 3D"):
        """
        Grafica puntos en 3D, mostrando opcionalmente los centroides y agrupando puntos según asignaciones.
        :param datos: Lista de datos, donde cada elemento tiene al menos 3 valores (x, y, z) y posiblemente una etiqueta en la 4ª posición.
        :param centroides: (Opcional) Array de centroides a graficar.
        :param asignaciones: (Opcional) Lista/array con el cluster asignado para cada punto.
        :param titulo: Título del gráfico.
        """
        self.axes.clear()                              # Limpia el eje para una nueva gráfica
        self.axes.set_title(titulo)                      # Establece el título
        self.axes.set_xlabel("Hu[6]")             # Etiqueta para el eje X
        self.axes.set_ylabel("Hu[1]")             # Etiqueta para el eje Y
        self.axes.set_zlabel("Hu[0]")                     # Etiqueta para el eje Z

        if asignaciones is not None:
            # Si hay asignaciones, se agrupan los puntos por cluster
            asignaciones = np.array(asignaciones)
            for cluster_id in np.unique(asignaciones):
                # Filtra los puntos correspondientes a este cluster
                cluster_puntos = np.array(
                    [d[:3] for i, d in enumerate(datos) if asignaciones[i] == cluster_id]
                )
                if len(cluster_puntos) > 0:
                    self.axes.scatter(
                        cluster_puntos[:, 0], cluster_puntos[:, 1], cluster_puntos[:, 2],
                        label=f"Cluster {cluster_id}"
                    )
            # Si hay centroides, se grafican con marcador 'X' de color negro
            if centroides is not None:
                self.axes.scatter(
                    centroides[:, 0], centroides[:, 1], centroides[:, 2],
                    c='black', marker='X', s=100, label="Centroides"
                )
        else:
            # Si no se proporcionan asignaciones, agrupa puntos según su etiqueta (en la posición 4)
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

        # Muestra la leyenda fuera del gráfico para mayor claridad
        self.axes.legend(
            loc='upper left', 
            bbox_to_anchor=(1.05, 1.0), 
            borderaxespad=0.
        )
        self.draw()                                    # Redibuja el canvas con los cambios

    def on_scroll(self, event):
        """Callback para manejar el zoom con la rueda del mouse en el gráfico 3D."""
        base_scale = 1.1                             # Factor base de zoom
        # Obtiene los límites actuales de cada eje en 3D
        x_left, x_right = self.axes.get_xlim3d()
        y_left, y_right = self.axes.get_ylim3d()
        z_left, z_right = self.axes.get_zlim3d()
        
        x_range = x_right - x_left                   # Rango en el eje X
        y_range = y_right - y_left                   # Rango en el eje Y
        z_range = z_right - z_left                   # Rango en el eje Z

        # Determina el factor de escala según la dirección del scroll:
        # 'up' para acercar (zoom in), 'down' para alejar (zoom out)
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        # Calcula el punto central de cada eje
        x_mid = (x_left + x_right) * 0.5
        y_mid = (y_left + y_right) * 0.5
        z_mid = (z_left + z_right) * 0.5

        # Calcula los nuevos rangos escalados
        new_x_range = x_range * scale_factor
        new_y_range = y_range * scale_factor
        new_z_range = z_range * scale_factor

        # Ajusta los límites de los ejes centrados en el punto medio
        self.axes.set_xlim3d([x_mid - new_x_range/2, x_mid + new_x_range/2])
        self.axes.set_ylim3d([y_mid - new_y_range/2, y_mid + new_y_range/2])
        self.axes.set_zlim3d([z_mid - new_z_range/2, z_mid + new_z_range/2])
        self.draw()                                    # Redibuja el gráfico con los nuevos límites
