from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QWheelEvent

class ZoomableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.pixmap_original = None
        self.setAlignment(Qt.AlignCenter)
        # Es recomendable que el contenido se ajuste al tamaño del label
        self.setScaledContents(False)

    def setPixmap(self, pixmap: QPixmap):
        """Guarda el pixmap original y lo muestra escalado según el zoom_factor."""
        self.pixmap_original = pixmap
        scaled = self.pixmap_original.scaled(
            self.pixmap_original.size() * self.zoom_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        super().setPixmap(scaled)

    def wheelEvent(self, event: QWheelEvent):
        """Aumenta o reduce el zoom en respuesta a la rueda del mouse."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.1
        elif delta < 0:
            self.zoom_factor /= 1.1
        if self.pixmap_original:
            scaled = self.pixmap_original.scaled(
                self.pixmap_original.size() * self.zoom_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)
