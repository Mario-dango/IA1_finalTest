from PyQt5.QtWidgets import QWidget             # Importa la clase QWidget para crear widgets personalizados
from PyQt5.QtGui import QPainter, QColor, QPen    # Importa QPainter para dibujar, QColor para definir colores y QPen para configurar trazos
from PyQt5.QtCore import Qt, QRect              # Importa constantes de Qt y QRect para definir rectángulos

class GridAstarWidget(QWidget):
    """
    Widget interactivo para el escenario de A*.
    La grilla es una matriz (lista de listas) donde:
      - 0 indica casilla transitable.
      - 1 indica obstáculo (ej. un estante, se pinta de amarillo).
    Permite seleccionar:
      - Primer click: posición de origen (se pinta en verde).
      - Segundo click: posición de destino (se pinta en rojo).
      - Tercer click: reinicia la selección.
    La ruta calculada se dibuja en azul.
    """
    def __init__(self, grid, cell_size=40, parent=None):
        super().__init__(parent)                        # Inicializa la clase base QWidget
        self.grid = grid                                # Guarda la grilla pasada como parámetro
        self.cell_size = cell_size                      # Define el tamaño de cada celda
        self.start = None                               # Inicializa la posición de inicio (origen) en None
        self.goal = None                                # Inicializa la posición de destino en None
        self.path = []                                  # Lista que almacenará la ruta calculada (como tuplas de (fila, columna))
        rows = len(grid)                                # Número de filas en la grilla
        cols = len(grid[0]) if rows > 0 else 0            # Número de columnas (si existen filas)
        self.setFixedSize(cols * self.cell_size, rows * self.cell_size)  # Fija el tamaño del widget según la grilla y el tamaño de celda

    def paintEvent(self, event):
        painter = QPainter(self)                        # Crea un objeto QPainter para dibujar en el widget
        rows = len(self.grid)                           # Obtiene el número de filas de la grilla
        cols = len(self.grid[0])                         # Obtiene el número de columnas de la grilla
        # Dibuja cada celda de la grilla y, si corresponde, los obstáculos
        for i in range(rows):
            for j in range(cols):
                # Define un rectángulo para la celda actual basado en su posición y tamaño
                rect = QRect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                # Si el valor en la grilla es 1 (obstáculo), se rellena con color amarillo
                if self.grid[i][j] == 1:
                    painter.fillRect(rect, QColor("yellow"))
                painter.drawRect(rect)                  # Dibuja el contorno del rectángulo
        # Dibuja la posición de origen (si fue seleccionada) en color verde
        if self.start is not None:
            i, j = self.start                           # Extrae las coordenadas del nodo de inicio
            rect = QRect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
            painter.fillRect(rect, QColor("green"))
        # Dibuja la posición de destino (si fue seleccionada) en color rojo
        if self.goal is not None:
            i, j = self.goal                            # Extrae las coordenadas del nodo de destino
            rect = QRect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
            painter.fillRect(rect, QColor("red"))
        # Si existe una ruta calculada, la dibuja en color azul
        if self.path:
            pen = QPen(QColor("blue"), 3)              # Crea un QPen azul con grosor 3
            painter.setPen(pen)                         # Asigna el pen al painter
            # Itera sobre la ruta para dibujar líneas entre cada par consecutivo de puntos
            for idx in range(1, len(self.path)):
                r1, c1 = self.path[idx - 1]             # Coordenadas del punto anterior
                r2, c2 = self.path[idx]                 # Coordenadas del punto actual
                # Calcula las coordenadas centrales de cada celda y las convierte a enteros
                x1 = int(c1 * self.cell_size + self.cell_size / 2)
                y1 = int(r1 * self.cell_size + self.cell_size / 2)
                x2 = int(c2 * self.cell_size + self.cell_size / 2)
                y2 = int(r2 * self.cell_size + self.cell_size / 2)
                painter.drawLine(x1, y1, x2, y2)         # Dibuja la línea entre los dos puntos

    def mousePressEvent(self, event):
        # Obtiene las coordenadas del clic
        x = event.x()
        y = event.y()
        # Calcula la columna y fila basadas en el tamaño de celda
        col = x // self.cell_size
        row = y // self.cell_size
        if event.button() == Qt.LeftButton:           # Solo responde al clic izquierdo
            if self.start is None:
                self.start = (row, col)               # Primer clic: asigna la posición de inicio
            elif self.goal is None:
                self.goal = (row, col)                # Segundo clic: asigna la posición de destino
            else:
                # Si ambos ya están definidos, reinicia la selección con el nuevo clic asignado a inicio
                self.start = (row, col)
                self.goal = None
                self.path = []                        # Limpia la ruta calculada
        self.update()                                   # Solicita actualizar/redibujar el widget
