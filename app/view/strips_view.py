# view/strips_view.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QPushButton, QTextEdit, QListWidgetItem
from PyQt5.QtCore import Qt

class StripsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Mostrar el orden actual de cajas (solo lectura)
        self.label_current = QLabel("Orden actual de cajas (detectado):")
        layout.addWidget(self.label_current)
        self.list_current = QListWidget()
        self.list_current.setSelectionMode(QListWidget.NoSelection)
        self.list_current.setDragDropMode(QListWidget.NoDragDrop)
        layout.addWidget(self.list_current)
        
        # Permitir que el usuario defina el orden objetivo (editable mediante drag & drop)
        self.label_goal = QLabel("Orden objetivo (arrastra para reordenar):")
        layout.addWidget(self.label_goal)
        self.list_goal = QListWidget()
        self.list_goal.setDragDropMode(QListWidget.InternalMove)
        layout.addWidget(self.list_goal)
        
        # Botón para calcular el plan STRIPS
        self.button_plan = QPushButton("Calcular Plan STRIPS")
        layout.addWidget(self.button_plan)
        
        # Área de texto para mostrar el plan resultante
        self.text_plan = QTextEdit()
        self.text_plan.setReadOnly(True)
        layout.addWidget(self.text_plan)
    
    def set_current_order(self, order):
        """Actualiza la lista de orden actual (solo lectura)."""
        self.list_current.clear()
        for caja in order:
            item = QListWidgetItem(caja)
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.list_current.addItem(item)
    
    def set_goal_order(self, order):
        """Inicializa la lista del orden objetivo con el orden dado."""
        self.list_goal.clear()
        for caja in order:
            self.list_goal.addItem(caja)
    
    def get_goal_order(self):
        """Retorna el orden objetivo actual (según lo reordenado por el usuario)."""
        order = []
        for i in range(self.list_goal.count()):
            order.append(self.list_goal.item(i).text())
        return order
    
    def display_plan(self, plan):
        """Muestra el plan STRIPS en el área de texto."""
        self.text_plan.clear()
        for action in plan:
            self.text_plan.append(action)
