def heuristic(node, goal):
    # Cálculo de la heurística utilizando distancia de Manhattan
    h_cost = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    return h_cost

def a_star(start, goal):
    open_list = PriorityQueue()
    open_list.put((0, start))  # Tupla: (f_cost, h_cost, nodo)
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_list.empty():
        current = open_list.get()[1]

        if current == goal:
            return reconstruct_path(goal)

        for neighbor in get_neighbors(current):
            temp_g_score = g_score[current] + 1

            if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor, goal)
                open_list.put((f_score[neighbor], neighbor))

    return None

start_cell = (15, 20)
goal_cell = (100, 100)
path = a_star(start_cell, goal_cell)