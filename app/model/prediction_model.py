import numpy as np

class PredictionModel:
    def __init__(self, database_model):
        """
        Inicializa el modelo de predicción.
        """
        self.centroides = None  # Centroides para K-means
        self.database_model = database_model  # Instancia de DatabaseModel para normalización

    def knn_manual(self, nuevo_punto, k=3):
        """
        Implementación manual de KNN usando las 3 características (normalizadas).
        Retorna la etiqueta predicha.
        """
        if not hasattr(self.database_model, 'datos_entrenamiento') or not self.database_model.datos_entrenamiento:
            return "Desconocido"  # Si no hay datos de entrenamiento

        nuevo_norm = self.database_model.normalizar(nuevo_punto)
        distancias = []
        for fila in self.database_model.datos_entrenamiento:
            features = fila[:3]
            features_norm = self.database_model.normalizar(features)
            dist = np.linalg.norm(nuevo_norm - features_norm)
            distancias.append((dist, fila[3]))  # (distancia, etiqueta)
        
        # Ordenar por distancia y seleccionar los k vecinos más cercanos
        distancias.sort(key=lambda x: x[0])
        vecinos = distancias[:k]

        # Votación mayoritaria
        conteo = {}
        for d, etiq in vecinos:
            conteo[etiq] = conteo.get(etiq, 0) + 1
        etiqueta_ganadora = max(conteo, key=conteo.get)
        return etiqueta_ganadora

    def kmeans_manual(self, datos, k=4, max_iter=100, n_init=10):
        """
        Implementa manualmente el algoritmo K-means sobre datos (lista de [circ, asp, hu0]).
        Se aplica una inicialización k-means++ y se realizan n_init reinicios para elegir la mejor solución.
        Retorna:
          - centroides (en el espacio NORMALIZADO)
          - asignaciones (para cada punto)
        """
        best_inertia = float('inf')               # Inercia mínima encontrada (suma de distancias cuadráticas)
        best_centroids = None                     # Centroides de la mejor solución
        best_asignaciones = None                  # Asignaciones de la mejor solución
        
        # Convierte cada dato a su versión normalizada (usando las 3 características)
        datos_np = np.array([self.database_model.normalizar(d) for d in datos], dtype=np.float32)
        N = datos_np.shape[0]
        if N < k:
            k = N  # Asegurarse de no tener más clusters que puntos
        
        for init in range(n_init):
            # Inicialización k-means++:
            centroids = []
            # Selecciona el primer centro de forma aleatoria
            first_idx = np.random.choice(N)
            centroids.append(datos_np[first_idx])
            # Selecciona los siguientes centros con probabilidad proporcional a la distancia al centro más cercano
            for _ in range(1, k):
                dist_sq = np.array([min([np.linalg.norm(p - c)**2 for c in centroids]) for p in datos_np])
                probs = dist_sq / dist_sq.sum()
                next_idx = np.random.choice(N, p=probs)
                centroids.append(datos_np[next_idx])
            centroids = np.array(centroids)
            
            # Iteración de K-means
            for i in range(max_iter):
                # Asigna cada punto al centro más cercano
                asignaciones = np.array([np.argmin([np.linalg.norm(p - c) for c in centroids]) for p in datos_np])
                nuevos_centroides = []
                # Recalcula cada centro como la media de los puntos asignados
                for cluster_id in range(k):
                    cluster_puntos = datos_np[asignaciones == cluster_id]
                    if len(cluster_puntos) > 0:
                        nuevo_c = np.mean(cluster_puntos, axis=0)
                    else:
                        # Si el cluster quedó vacío, se reinicializa con un punto aleatorio
                        nuevo_c = datos_np[np.random.choice(N)]
                    nuevos_centroides.append(nuevo_c)
                nuevos_centroides = np.array(nuevos_centroides)
                # Si los centros no cambian significativamente, se asume convergencia
                if np.allclose(centroids, nuevos_centroides, atol=1e-6):
                    break
                centroids = nuevos_centroides
            
            # Calcula la inercia (suma de distancias cuadráticas de cada punto a su centro)
            inertia = sum([np.linalg.norm(datos_np[i] - centroids[asignaciones[i]])**2 for i in range(N)])
            # Guarda la solución si mejora la inercia
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_asignaciones = asignaciones
        
        self.centroides = best_centroids  # Guarda los centroides del mejor reinicio (en el espacio normalizado)
        return best_centroids, best_asignaciones


    def get_centroids_unnorm(self):
        """
        Retorna los centroides en el espacio original (desnormalizados).
        """
        if self.centroides is None:
            return None
        centroids_unnorm = []
        for c in self.centroides:
            c_desnorm = self.database_model.desnormalizar(c)
            centroids_unnorm.append(c_desnorm)
        return np.array(centroids_unnorm)