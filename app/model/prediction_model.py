import numpy as np  # Importa NumPy para cálculos y manejo de arrays

class PredictionModel:
    def __init__(self, database_model):
        """
        Inicializa el modelo de predicción.
        :param database_model: Instancia de DatabaseModel, necesaria para normalización.
        """
        self.centroides = None               # Almacenará los centroides obtenidos en K-means (espacio normalizado)
        self.database_model = database_model  # Guarda la instancia de DatabaseModel para usar sus métodos (normalización)

    def knn_manual(self, nuevo_punto, k=3):
        """
        Implementación manual del algoritmo k-NN usando 3 características (normalizadas).
        :param nuevo_punto: Vector de características del nuevo punto.
        :param k: Número de vecinos a considerar.
        :return: Etiqueta predicha.
        """
        # Verifica si existen datos de entrenamiento; de lo contrario, retorna "Desconocido"
        if not hasattr(self.database_model, 'datos_entrenamiento') or not self.database_model.datos_entrenamiento:
            return "Desconocido"
        # Normaliza el nuevo punto utilizando la media y std del dataset
        nuevo_norm = self.database_model.normalizar(nuevo_punto)
        distancias = []
        # Recorre cada muestra del dataset
        for fila in self.database_model.datos_entrenamiento:
            features = fila[:3]  # Toma las 3 primeras características
            features_norm = self.database_model.normalizar(features)
            dist = np.linalg.norm(nuevo_norm - features_norm)  # Calcula la distancia euclidiana
            distancias.append((dist, fila[3]))  # Guarda la distancia y la etiqueta asociada
        # Ordena los vecinos por distancia (de menor a mayor)
        distancias.sort(key=lambda x: x[0])
        vecinos = distancias[:k]  # Selecciona los k vecinos más cercanos

        # Realiza una votación mayoritaria para determinar la etiqueta final
        conteo = {}
        for d, etiq in vecinos:
            conteo[etiq] = conteo.get(etiq, 0) + 1
        etiqueta_ganadora = max(conteo, key=conteo.get)
        return etiqueta_ganadora

    def kmeans_manual(self, datos, k=4, max_iter=100):
        """
        Implementa manualmente el algoritmo K-means sobre datos.
        :param datos: Lista de datos, donde cada elemento es [circularidad, aspect_ratio, hu0].
        :param k: Número de clusters deseados.
        :param max_iter: Número máximo de iteraciones para la convergencia.
        :return: Tuple (centroides, asignaciones) en el espacio normalizado.
        """
        # Normaliza cada dato usando el método de DatabaseModel
        datos_np = np.array([self.database_model.normalizar(d) for d in datos], dtype=np.float32)
        N = datos_np.shape[0]
        if N < k:
            k = N  # No se pueden tener más clusters que puntos

        # Inicializa los centroides seleccionando k puntos aleatorios sin reemplazo
        idx_random = np.random.choice(N, k, replace=False)
        centroides = datos_np[idx_random, :]

        for _ in range(max_iter):
            asignaciones = []
            # Para cada punto, calcula la distancia a cada centroide y asigna al cluster más cercano
            for punto in datos_np:
                distancias = [np.linalg.norm(punto - c) for c in centroides]
                cluster_id = np.argmin(distancias)
                asignaciones.append(cluster_id)
            asignaciones = np.array(asignaciones)

            # Actualiza los centroides como la media de los puntos asignados a cada cluster
            nuevos_centroides = []
            for cluster_id in range(k):
                cluster_puntos = datos_np[asignaciones == cluster_id]
                if len(cluster_puntos) > 0:
                    nuevo_c = np.mean(cluster_puntos, axis=0)
                else:
                    # Si un cluster queda vacío, reinicializa con un punto aleatorio
                    nuevo_c = datos_np[np.random.choice(N)]
                nuevos_centroides.append(nuevo_c)
            nuevos_centroides = np.array(nuevos_centroides)

            # Verifica la convergencia: si los centroides no cambian, rompe el bucle
            if np.allclose(centroides, nuevos_centroides, atol=1e-6):
                break
            centroides = nuevos_centroides

        self.centroides = centroides  # Guarda los centroides finales en el espacio normalizado
        return centroides, asignaciones

    def get_centroids_unnorm(self):
        """
        Retorna los centroides guardados, desnormalizados al espacio original.
        :return: Array de centroides en el espacio original.
        """
        if self.centroides is None:
            return None
        centroids_unnorm = []
        for c in self.centroides:
            c_desnorm = self.database_model.desnormalizar(c)  # Desnormaliza cada centroide
            centroids_unnorm.append(c_desnorm)
        return np.array(centroids_unnorm)
