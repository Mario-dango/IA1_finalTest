import heapq  # Importa heapq para manejar la cola de prioridad (min-heap) que usa A*

class AStarModel:
    """
    Clase que encapsula la lógica del algoritmo A* aplicado a un grid.
    Se asume que el grid es una lista de listas, donde:
      - 0 indica una casilla transitable.
      - 1 indica un obstáculo.
    """
    def __init__(self, grid=None):
        # Inicializa la instancia con un grid opcional.
        self.grid = grid      # Guarda el grid (si se pasa alguno)
        self.rows = 0         # Inicializa el número de filas a 0
        self.cols = 0         # Inicializa el número de columnas a 0
        self.costs = None     # Inicializa la matriz de costos
        self.heuristic_mods = None  # Inicializa la matriz de modificadores de heurística
        if grid:              # Si se proporciona un grid...
            self.rows = len(grid)         # Asigna el número de filas
            self.cols = len(grid[0])      # Asigna el número de columnas (se asume que todas las filas tienen la misma longitud)
            # Inicializa la matriz de costos con 1 para cada celda
            self.costs = [[1 for _ in range(self.cols)] for _ in range(self.rows)]
            # Inicializa la matriz de modificadores de heurística con 0 para cada celda
            self.heuristic_mods = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def set_grid(self, grid):
        """
        Permite actualizar el grid de forma dinámica.
        :param grid: Nueva grilla (lista de listas).
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.costs = [[1 for _ in range(self.cols)] for _ in range(self.rows)]
        self.heuristic_mods = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def heuristic(self, a, b):
        """
        Calcula la heurística Manhattan entre dos nodos.
        :param a: Tupla (fila, columna) del nodo a.
        :param b: Tupla (fila, columna) del nodo b.
        :return: Distancia Manhattan entre a y b.
        """
        (x1, y1) = a   # Extrae las coordenadas del nodo a
        (x2, y2) = b   # Extrae las coordenadas del nodo b
        # Suma de las diferencias absolutas más el modificador de heurística
        return abs(x1 - x2) + abs(y1 - y2) + self.heuristic_mods[x1][y1]

    def get_neighbors(self, node):
        """
        Retorna los vecinos válidos (movimientos: abajo, arriba, derecha, izquierda)
        que sean transitables (valor 0) y estén dentro del grid.
        :param node: Tupla (fila, columna) del nodo actual.
        :return: Lista de vecinos válidos.
        """
        (x, y) = node              # Desempaqueta las coordenadas del nodo
        neighbors = []             # Lista para almacenar vecinos
        # Define las posibles direcciones de movimiento
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy  # Calcula la posición del vecino
            if 0 <= nx < self.rows and 0 <= ny < self.cols:  # Verifica que esté dentro de los límites
                if self.grid[nx][ny] == 0:  # Si la celda es transitable (0)
                    neighbors.append((nx, ny))  # Agrega el vecino a la lista
        return neighbors

    def set_cost(self, row, col, cost):
        """
        Establece el costo de una celda específica.
        :param row: Fila de la celda.
        :param col: Columna de la celda.
        :param cost: El nuevo costo a asignar.
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.costs[row][col] = cost

    def set_heuristic_mod(self, row, col, value):
        """
        Establece el modificador de heurística de una celda específica.
        :param row: Fila de la celda.
        :param col: Columna de la celda.
        :param value: El nuevo valor a asignar.
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.heuristic_mods[row][col] = value

    def reset(self):
        """
        Restablece los costos y modificadores de heurística a sus valores por defecto.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                self.costs[r][c] = 1
                self.heuristic_mods[r][c] = 0

    def astar_search(self, start, goal):
        """
        Ejecuta el algoritmo A* para encontrar la ruta más corta desde start a goal.
        :param start: Tupla (fila, columna) de inicio.
        :param goal: Tupla (fila, columna) de meta.
        :return: Lista de nodos (tuplas) que forman el camino o None si no se encuentra camino.
        """
        open_set = []  # Cola de prioridad para nodos por explorar
        heapq.heappush(open_set, (0, start))  # Inserta el nodo de inicio con prioridad 0

        came_from = {}  # Diccionario para reconstruir el camino: nodo actual -> nodo previo
        g_score = {start: 0}  # g_score almacena el costo desde el inicio hasta cada nodo
        f_score = {start: self.heuristic(start, goal)}  # f_score es g_score + heurística

        while open_set:  # Mientras haya nodos por explorar...
            _, current = heapq.heappop(open_set)  # Extrae el nodo con menor f_score

            if current == goal:  # Si se ha alcanzado la meta...
                return self.reconstruct_path(came_from, current)  # Reconstruye y retorna el camino

            # Itera por cada vecino del nodo actual
            for neighbor in self.get_neighbors(current):
                # Costo tentativo al vecino (costo del vecino)
                tentative_g = g_score[current] + self.costs[neighbor[0]][neighbor[1]]
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current  # Registra que para llegar a 'neighbor' se pasó por 'current'
                    g_score[neighbor] = tentative_g  # Actualiza el costo real para 'neighbor'
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)  # Actualiza el costo estimado
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))  # Inserta el vecino en la cola de prioridad

        return None  # Si se termina la exploración sin alcanzar la meta, retorna None

    def reconstruct_path(self, came_from, current):
        """
        Reconstruye la ruta a partir del diccionario came_from.
        :param came_from: Diccionario con la relación nodo actual -> nodo previo.
        :param current: Nodo final (meta) desde donde iniciar la reconstrucción.
        :return: Lista de nodos que forman el camino desde el inicio hasta el objetivo.
        """
        path = [current]  # Inicializa la ruta con el nodo final
        while current in came_from:
            current = came_from[current]  # Retrocede al nodo anterior
            path.append(current)          # Agrega el nodo a la ruta
        path.reverse()  # Invierte la lista para que comience en el nodo de inicio
        return path
