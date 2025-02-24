import heapq

class AStarModel:
    """
    Clase que encapsula la lógica de A* sobre un grid.
    El grid se asume como una lista de listas,
    donde 0 indica casilla transitable y 1 indica obstáculo.
    """
    def __init__(self, grid=None):
        # Almacena el grid y sus dimensiones
        self.grid = grid
        self.rows = 0
        self.cols = 0
        if grid:
            self.rows = len(grid)
            self.cols = len(grid[0])

    def set_grid(self, grid):
        """
        Permite setear o actualizar el grid de manera dinámica.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def heuristic(self, a, b):
        """
        Heurística Manhattan para A*.
        a y b son tuplas (fila, columna).
        """
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)

    def get_neighbors(self, node):
        """
        Retorna los vecinos válidos (arriba, abajo, izq, der)
        que no sean obstáculos y estén dentro del grid.
        """
        (x, y) = node
        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                # Verifica que no sea obstáculo
                if self.grid[nx][ny] == 0:
                    neighbors.append((nx, ny))
        return neighbors

    def astar_search(self, start, goal):
        """
        Ejecuta el algoritmo A* en el grid almacenado.
        Retorna la ruta como lista de nodos (fila, col) si existe,
        o None si no hay camino.
        """
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}  # Para reconstruir el camino
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            # Si llegamos a la meta, reconstruimos la ruta
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1  # coste de cada paso = 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No se encontró ruta
        return None

    def reconstruct_path(self, came_from, current):
        """
        Reconstruye la ruta desde 'start' hasta 'current' usando el diccionario 'came_from'.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
