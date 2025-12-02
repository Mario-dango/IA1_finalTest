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

    def kmeans_manual(self, datos, k=4, max_iter=100):
        """
        Implementación manual de K-means sobre datos: lista de [circ, aspect_ratio, hu0].
        Retorna:
          - centroides (en el espacio NORMALIZADO)
          - asignaciones (para cada punto)
        """
        datos_np = np.array([self.database_model.normalizar(d) for d in datos], dtype=np.float32)
        N = datos_np.shape[0]
        if N < k:
            k = N  # No podemos tener más clusters que puntos

        # Inicializar centroides aleatoriamente
        idx_random = np.random.choice(N, k, replace=False)
        centroides = datos_np[idx_random, :]

        for _ in range(max_iter):
            asignaciones = []
            for punto in datos_np:
                distancias = [np.linalg.norm(punto - c) for c in centroides]
                cluster_id = np.argmin(distancias)
                asignaciones.append(cluster_id)
            asignaciones = np.array(asignaciones)

            # Actualizar centroides
            nuevos_centroides = []
            for cluster_id in range(k):
                cluster_puntos = datos_np[asignaciones == cluster_id]
                if len(cluster_puntos) > 0:
                    nuevo_c = np.mean(cluster_puntos, axis=0)
                else:
                    nuevo_c = datos_np[np.random.choice(N)]  # Si un cluster está vacío, reinicializar
                nuevos_centroides.append(nuevo_c)
            nuevos_centroides = np.array(nuevos_centroides)

            # Verificar convergencia
            if np.allclose(centroides, nuevos_centroides, atol=1e-6):
                break
            centroides = nuevos_centroides

        self.centroides = centroides  # Guardar centroides en el espacio normalizado
        return centroides, asignaciones

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