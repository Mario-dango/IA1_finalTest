# view/strips_view.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QPushButton, QTextEdit, QListWidgetItem
from PyQt5.QtCore import Qt

class StripsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)           # Inicializa el widget base
        self.init_ui()                     # Llama al método que configura la interfaz

    def init_ui(self):
        layout = QVBoxLayout(self)         # Crea un layout vertical para organizar los elementos
        
        # Se crea y añade un QLabel para mostrar el orden actual de las cajas (solo lectura)
        self.label_current = QLabel("Orden actual de cajas (detectado):")
        layout.addWidget(self.label_current)
        # Se crea un QListWidget para listar el orden actual; no se permite selección ni drag & drop
        self.list_current = QListWidget()
        self.list_current.setSelectionMode(QListWidget.NoSelection)
        self.list_current.setDragDropMode(QListWidget.NoDragDrop)
        layout.addWidget(self.list_current)
        
        # Se crea y añade un QLabel para mostrar el orden objetivo (editable mediante drag & drop)
        self.label_goal = QLabel("Orden objetivo (arrastra para reordenar):")
        layout.addWidget(self.label_goal)
        # Se crea un QListWidget para permitir al usuario reordenar el orden objetivo internamente
        self.list_goal = QListWidget()
        self.list_goal.setDragDropMode(QListWidget.InternalMove)
        layout.addWidget(self.list_goal)
        
        # Se crea un botón para calcular el plan STRIPS y se añade al layout
        self.button_plan = QPushButton("Calcular Plan STRIPS")
        layout.addWidget(self.button_plan)
        
        # Se crea un QTextEdit de solo lectura para mostrar el plan resultante
        self.text_plan = QTextEdit()
        self.text_plan.setReadOnly(True)
        layout.addWidget(self.text_plan)
    
    def set_current_order(self, order):
        """
        Actualiza el QListWidget que muestra el orden actual de cajas.
        :param order: Lista de nombres de cajas en el orden actual.
        """
        self.list_current.clear()           # Limpia la lista actual
        for caja in order:
            item = QListWidgetItem(caja)    # Crea un item para cada caja
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)  # Desactiva la selección del item
            self.list_current.addItem(item) # Añade el item al QListWidget
    
    def set_goal_order(self, order):
        """
        Inicializa el QListWidget para el orden objetivo con la lista dada.
        :param order: Lista de nombres de cajas en el orden deseado.
        """
        self.list_goal.clear()              # Limpia la lista existente
        for caja in order:
            self.list_goal.addItem(caja)     # Añade cada caja como item editable por el usuario
    
    def get_goal_order(self):
        """
        Retorna el orden actual del QListWidget de orden objetivo.
        :return: Lista de nombres de cajas según el orden actual en el widget.
        """
        order = []
        for i in range(self.list_goal.count()):
            order.append(self.list_goal.item(i).text())
        return order
    
    def display_plan(self, plan):
        """
        Muestra la secuencia de acciones (plan STRIPS) en el QTextEdit.
        :param plan: Lista de acciones (cadenas de texto).
        """
        self.text_plan.clear()              # Limpia el área de texto
        for action in plan:
            self.text_plan.append(action)   # Añade cada acción al QTextEdit
