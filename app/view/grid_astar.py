from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect

class GridAstarWidget(QWidget):
    """
    Widget interactivo para el escenario de A*.
    La grilla es una matriz (lista de listas) donde:
      - 0 indica celda transitable.
      - 1 indica obstáculo (por ejemplo, un estante, se pinta de amarillo).
    Permite seleccionar:
      - Primer click: posición de origen (verde).
      - Segundo click: posición de destino (rojo).
      - Tercer click: reinicia la selección.
    La ruta calculada se dibuja en azul.
    """
    def __init__(self, grid, cell_size=40, parent=None):
        super().__init__(parent)
        self.grid = grid
        self.cell_size = cell_size
        self.start = None   # (fila, columna)
        self.goal = None    # (fila, columna)
        self.path = []      # Ruta calculada (lista de (fila, columna))
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        self.setFixedSize(cols * self.cell_size, rows * self.cell_size)

    def paintEvent(self, event):
        painter = QPainter(self)
        rows = len(self.grid)
        cols = len(self.grid[0])
        # Dibujar celdas y obstáculos
        for i in range(rows):
            for j in range(cols):
                rect = QRect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                if self.grid[i][j] == 1:
                    painter.fillRect(rect, QColor("yellow"))
                painter.drawRect(rect)
        # Dibujar posición de origen (verde)
        if self.start is not None:
            i, j = self.start
            rect = QRect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
            painter.fillRect(rect, QColor("green"))
        # Dibujar posición de destino (rojo)
        if self.goal is not None:
            i, j = self.goal
            rect = QRect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
            painter.fillRect(rect, QColor("red"))
        # Dibujar la ruta (azul)
        if self.path:
            pen = QPen(QColor("blue"), 3)
            painter.setPen(pen)
            for idx in range(1, len(self.path)):
                r1, c1 = self.path[idx - 1]
                r2, c2 = self.path[idx]
                x1 = int(c1 * self.cell_size + self.cell_size / 2)
                y1 = int(r1 * self.cell_size + self.cell_size / 2)
                x2 = int(c2 * self.cell_size + self.cell_size / 2)
                y2 = int(r2 * self.cell_size + self.cell_size / 2)
                painter.drawLine(x1, y1, x2, y2)


    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        col = x // self.cell_size
        row = y // self.cell_size
        if event.button() == Qt.LeftButton:
            if self.start is None:
                self.start = (row, col)
            elif self.goal is None:
                self.goal = (row, col)
            else:
                # Reinicia la selección: nuevo origen, sin destino ni ruta.
                self.start = (row, col)
                self.goal = None
                self.path = []
        self.update()
