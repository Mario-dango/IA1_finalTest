from PyQt5.QtWidgets import QLabel             # Importa QLabel para mostrar imágenes
from PyQt5.QtCore import Qt                    # Importa constantes de Qt (por ejemplo, AlignCenter)
from PyQt5.QtGui import QPixmap, QWheelEvent     # Importa QPixmap para imágenes y QWheelEvent para eventos de la rueda del mouse

class ZoomableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)                  # Inicializa la clase base QLabel
        self.zoom_factor = 1.0                    # Factor de zoom inicial
        self.pixmap_original = None               # Variable para almacenar el pixmap original sin escalado
        self.setAlignment(Qt.AlignCenter)         # Centra el contenido dentro del QLabel
        # Se establece que el contenido no se escala automáticamente (se hará de forma manual)
        self.setScaledContents(False)

    def setPixmap(self, pixmap: QPixmap):
        """
        Sobrescribe el método setPixmap para guardar el pixmap original y mostrarlo escalado según el zoom_factor.
        :param pixmap: Objeto QPixmap que se desea mostrar.
        """
        self.pixmap_original = pixmap            # Guarda la imagen original
        scaled = self.pixmap_original.scaled(
            self.pixmap_original.size() * self.zoom_factor,  # Escala el tamaño según el factor de zoom
            Qt.KeepAspectRatio,                  # Mantiene la relación de aspecto
            Qt.SmoothTransformation              # Aplica una transformación suave para una mejor calidad
        )
        super().setPixmap(scaled)                # Llama al método setPixmap de la clase base para mostrar la imagen escalada

    def wheelEvent(self, event: QWheelEvent):
        """
        Maneja el evento de la rueda del mouse para ajustar el zoom.
        :param event: QWheelEvent con información del scroll.
        """
        delta = event.angleDelta().y()           # Obtiene el cambio en el ángulo del scroll en el eje Y
        if delta > 0:
            self.zoom_factor *= 1.1              # Si el scroll es hacia arriba, aumenta el zoom
        elif delta < 0:
            self.zoom_factor /= 1.1              # Si es hacia abajo, reduce el zoom
        if self.pixmap_original:                 # Si hay una imagen original cargada
            scaled = self.pixmap_original.scaled(
                self.pixmap_original.size() * self.zoom_factor,  # Calcula el nuevo tamaño
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)            # Actualiza la imagen mostrada en el QLabel
