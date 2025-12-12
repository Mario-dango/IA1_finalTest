import numpy as np

class PredictionModel:
    def __init__(self, database_model):
        self.centroides = None
        self.database_model = database_model

    def knn_manual(self, nuevo_punto, k=3):
        if not hasattr(self.database_model, 'datos_entrenamiento') or not self.database_model.datos_entrenamiento:
            return "Desconocido"

        nuevo_norm = self.database_model.normalizar(nuevo_punto)
        distancias = []
        for fila in self.database_model.datos_entrenamiento:
            features = fila[:3]
            features_norm = self.database_model.normalizar(features)
            dist = np.linalg.norm(nuevo_norm - features_norm)
            distancias.append((dist, fila[3]))
        
        distancias.sort(key=lambda x: x[0])
        vecinos = distancias[:k]

        conteo = {}
        for d, etiq in vecinos:
            conteo[etiq] = conteo.get(etiq, 0) + 1
        if conteo:
            etiqueta_ganadora = max(conteo, key=conteo.get)
            return etiqueta_ganadora
        return "Desconocido"

    def kmeans_manual(self, datos, k=4, max_iter=100, centroides_iniciales=None):
        """
        K-means manual.
        :param centroides_iniciales: Lista de vectores NORMALIZADOS para iniciar los centroides.
        """
        # Convertir datos a numpy y normalizar
        datos_np = np.array([self.database_model.normalizar(d) for d in datos], dtype=np.float32)
        N = datos_np.shape[0]
        
        # --- Lógica de Inicialización ---
        if centroides_iniciales is not None and len(centroides_iniciales) == k:
            # Usar centroides forzados (para que Cluster 0 sea siempre la primera categoría, etc.)
            centroides = np.array(centroides_iniciales, dtype=np.float32)
        else:
            # Inicialización aleatoria (fallback)
            if N < k:
                k = N
            idx_random = np.random.choice(N, k, replace=False)
            centroides = datos_np[idx_random, :]
        
        # --- Bucle principal ---
        for _ in range(max_iter):
            asignaciones = []
            for punto in datos_np:
                # Calcular distancia a cada centroide
                distancias = [np.linalg.norm(punto - c) for c in centroides]
                cluster_id = np.argmin(distancias)
                asignaciones.append(cluster_id)
            asignaciones = np.array(asignaciones)

            nuevos_centroides = []
            cambio_significativo = False
            
            for cluster_id in range(k):
                puntos_cluster = datos_np[asignaciones == cluster_id]
                if len(puntos_cluster) > 0:
                    nuevo_c = np.mean(puntos_cluster, axis=0)
                else:
                    # Si un cluster muere, reiniciarlo en un punto aleatorio
                    nuevo_c = datos_np[np.random.choice(N)]
                
                nuevos_centroides.append(nuevo_c)
            
            nuevos_centroides = np.array(nuevos_centroides)
            
            # Verificar convergencia
            if np.allclose(centroides, nuevos_centroides, atol=1e-5):
                break
            centroides = nuevos_centroides

        self.centroides = centroides
        return centroides, asignaciones

    def get_centroids_unnorm(self):
        """ Retorna los centroides desnormalizados para poder graficarlos en la escala real. """
        if self.centroides is None:
            return None
        centroids_unnorm = []
        for c in self.centroides:
            c_desnorm = self.database_model.desnormalizar(c)
            centroids_unnorm.append(c_desnorm)
        return np.array(centroids_unnorm)