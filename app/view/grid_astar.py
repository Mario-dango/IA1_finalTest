from PyQt5.QtWidgets import QWidget, QInputDialog
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect

class GridAstarWidget(QWidget):
    """
    Widget interactivo para el escenario de A*.
    ...
    """
    def __init__(self, grid, cell_size=40, parent=None):
        super().__init__(parent)
        self.grid = grid
        self.costs = []
        self.cost_update_callback = None
        self.heuristic_mods = []
        self.heuristic_update_callback = None
        self.cell_size = cell_size
        self.start = None
        self.goal = None
        self.path = []
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        self.setFixedSize(cols * self.cell_size, rows * self.cell_size)

    def paintEvent(self, event):
        painter = QPainter(self)
        rows = len(self.grid)
        cols = len(self.grid[0])
        
        # Define el color del borde para TODA la grilla
        # Usamos 'white' para que resalte en tu tema oscuro
        painter.setPen(QColor("white")) 

        for i in range(rows):
            for j in range(cols):
                rect = QRect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                
                # --- Lógica de Pintado de Celdas (Relleno) ---
                if self.grid[i][j] == 1:
                    painter.fillRect(rect, QColor("yellow"))
                else:
                    cost = self.costs[i][j]
                    if cost > 1:
                        # Color más oscuro si el costo es alto
                        color_value = max(0, 255 - (cost - 1) * 20)
                        painter.fillRect(rect, QColor(color_value, color_value, 255))
                    
                    # Dibujar texto del costo
                    painter.drawText(rect, Qt.AlignCenter, str(cost))

                # Modificador heurístico (Texto rojo)
                if self.heuristic_mods and self.heuristic_mods[i][j] != 0:
                    painter.setPen(QColor("red"))
                    painter.drawText(rect, Qt.AlignTop | Qt.AlignRight, str(self.heuristic_mods[i][j]))
                    painter.setPen(QColor("white")) # Restaurar a blanco

                # Dibujar el rectángulo de la celda (Bordes internos)
                painter.drawRect(rect)

        # --- CORRECCIÓN: DIBUJAR MARCO FINAL DENTRO DEL ÁREA VISIBLE ---
        
        # Calculamos el límite máximo
        total_w = cols * self.cell_size
        total_h = rows * self.cell_size
        
        # IMPORTANTE: Restamos 1 pixel para que la línea quede DENTRO del canvas
        # Si dibujamos en 'total_w', quedamos fuera.
        x_limit = total_w - 1
        y_limit = total_h - 1

        painter.setPen(QColor("white"))
        
        # Línea vertical derecha (desde arriba hasta abajo)
        painter.drawLine(x_limit, 0, x_limit, y_limit)
        
        # Línea horizontal inferior (desde izquierda hasta derecha)
        painter.drawLine(0, y_limit, x_limit, y_limit)
        
        # ------------------------------------------------------------------
        
        if self.start is not None:
            i, j = self.start
            rect = QRect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
            painter.fillRect(rect, QColor("green"))
        
        if self.goal is not None:
            i, j = self.goal
            rect = QRect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
            painter.fillRect(rect, QColor("red"))
        
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

        if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0]):
            if event.button() == Qt.LeftButton:
                if self.grid[row][col] == 0:  # Solo si no es un obstáculo
                    if self.start is None:
                        self.start = (row, col)
                    elif self.goal is None:
                        self.goal = (row, col)
                    else:
                        self.start = (row, col)
                        self.goal = None
                        self.path = []
            elif event.button() == Qt.RightButton:
                if self.grid[row][col] == 0:  # Solo si no es un obstáculo
                    cost, ok = QInputDialog.getInt(self, "Nuevo Costo", f"Ingrese el costo para la celda ({row}, {col}):", self.costs[row][col], 1)
                    if ok:
                        self.cost_update_callback(row, col, cost)
            elif event.button() == Qt.MidButton:
                if self.grid[row][col] == 0:
                    value, ok = QInputDialog.getInt(self, "Nuevo Modificador de Heurística", f"Ingrese el modificador para la celda ({row}, {col}):", self.heuristic_mods[row][col], -100, 100, 1)
                    if ok and self.heuristic_update_callback:
                        self.heuristic_update_callback(row, col, value)
        
        self.update()
