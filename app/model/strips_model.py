# model/strips_model.py

class StripsModel:
    def __init__(self):
        pass

    def plan_reordering(self, initial_order, goal_order):
        """
        Genera un plan de reordenamiento de cajas usando un enfoque simplificado.
        Los estados se representan como listas de strings (de arriba a abajo).
        Ejemplo:
            initial_order = ["tornillos", "clavos", "tuercas", "arandelas"]
            goal_order    = ["arandelas", "tornillos", "tuercas", "clavos"]
        Retorna una lista de acciones (strings) que simulan el plan.
        """
        plan = []
        if initial_order == goal_order:
            plan.append("El orden actual es el mismo que el deseado. No se requiere acción.")
            return plan

        plan.append("Iniciar reordenamiento de cajas:")
        plan.append("1. Desapilar todas las cajas.")
        for idx, caja in enumerate(goal_order):
            plan.append(f"2.{idx+1} Colocar la caja '{caja}' en la posición {idx+1}.")
        plan.append("3. Apilar las cajas en el orden deseado.")
        return plan

    def get_domain_str(self):
        """
        Retorna una cadena con la definición del dominio STRIPS para reordenar cajas.
        """
        domain = """
(define (domain reordenamiento)
  (:requirements :strips)
  (:predicates 
    (on ?x ?y)
    (clear ?x)
    (table ?x)
  )
  (:action desapilar
    :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x))
    :effect (and (not (on ?x ?y)) (clear ?y) (table ?x))
  )
  (:action apilar
    :parameters (?x ?y)
    :precondition (and (clear ?x) (clear ?y) (table ?x))
    :effect (and (on ?x ?y) (not (clear ?y)) (not (table ?x)))
  )
)
"""
        return domain

    def get_problem_str(self, initial_order, goal_order):
        """
        Retorna una cadena con la definición del problema STRIPS,
        basándose en el orden actual (initial_order) y el orden deseado (goal_order).
        Se asume que las cajas se apilan de arriba a abajo.
        """
        # Convertir la lista de cajas en una cadena de objetos (asumimos tipo "caja")
        objects_str = " ".join(initial_order) + " - caja"
        problem = f"""
(define (problem reordenar-cajas)
  (:domain reordenamiento)
  (:objects {objects_str})
  (:init
"""
        # Suponemos que el primer elemento es el tope y el último está en la mesa.
        if len(initial_order) > 0:
            for i in range(len(initial_order) - 1):
                problem += f"    (on {initial_order[i]} {initial_order[i+1]})\n"
            problem += f"    (table {initial_order[-1]})\n"
            problem += f"    (clear {initial_order[0]})\n"
        problem += "  )\n"
        problem += "  (:goal (and\n"
        if len(goal_order) > 0:
            for i in range(len(goal_order) - 1):
                problem += f"    (on {goal_order[i]} {goal_order[i+1]})\n"
            problem += f"    (table {goal_order[-1]})\n"
            problem += f"    (clear {goal_order[0]})\n"
        problem += "  ))\n"
        problem += ")"
        return problem
