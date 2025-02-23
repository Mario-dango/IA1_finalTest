import os
import cv2
import numpy as np
import heapq
import random
from collections import Counter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk  # Requiere Pillow

# ======================
# Funciones para VISIÓN
# ======================

def extract_features(image_path):
    """
    Extrae un histograma de color en espacio HSV como vector de características.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo cargar la imagen: " + image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Histograma con 8 bins por canal (8x8x8 = 512 características)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def load_dataset(folder_path):
    """
    Recorre la carpeta 'dataset' con la siguiente estructura:
      dataset/
          tornillo/
          clavo/
          tuerca/
          arandela/
    y extrae las características de cada imagen.
    """
    data = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            print(f"Carpeta encontrada: {label_path}")  # <--- DEBUG
            for filename in os.listdir(label_path):
                image_path = os.path.join(label_path, filename)
                try:
                    features = extract_features(image_path)
                    data.append(features)
                    labels.append(label)
                    print(f"  -> {filename} agregado a la clase '{label}'")  # <--- DEBUG
                except Exception as e:
                    print("Error al procesar", image_path, ":", e)
    print(f"Total imágenes cargadas: {len(data)}")
    print(f"Labels: {set(labels)}")
    return np.array(data), labels

def knn_classify(test_sample, train_data, train_labels, k=3):
    """
    Clasifica una muestra usando KNN.
    """
    distances = np.linalg.norm(train_data - test_sample, axis=1)
    k_indices = distances.argsort()[:k]
    k_labels = [train_labels[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]

def kmeans(data, k, max_iters=100):
    """
    Implementa K-means para agrupar los vectores de características.
    """
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]
    
    for _ in range(max_iters):
        clusters = {j: [] for j in range(k)}
        # Asigna cada punto al centroide más cercano
        for point in data:
            distances = np.linalg.norm(point - centroids, axis=1)
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
        
        new_centroids = []
        for j in range(k):
            if clusters[j]:
                new_centroid = np.mean(clusters[j], axis=0)
            else:
                new_centroid = centroids[j]
            new_centroids.append(new_centroid)
        new_centroids = np.array(new_centroids)
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

def classify_kmeans(test_sample, centroids):
    """
    Asigna la muestra al cluster con el centroide más cercano.
    """
    distances = np.linalg.norm(centroids - test_sample, axis=1)
    return np.argmin(distances)

# =====================
# Funciones para A* 
# =====================

def a_star(grid, start, goal):
    """
    Implementación de A* para encontrar el camino en una grilla.
    grid: lista de listas donde 0 es celda libre y 1 obstáculo.
    start y goal: tuplas (fila, columna).
    """
    def heuristic(a, b):
        # Distancia Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    closed_set = set()
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in closed_set:
            continue
        closed_set.add(current)
        
        # Movimientos: arriba, abajo, izquierda, derecha
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < len(grid) and 
                0 <= neighbor[1] < len(grid[0]) and 
                grid[neighbor[0]][neighbor[1]] == 0):
                if neighbor in closed_set:
                    continue
                new_g = g + 1
                new_f = new_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))
    return None

# ================================
# Datos iniciales para STRIPS (PDDL)
# ================================
STRIPS_DOMAIN = """(define (domain reordenar_cajas)
  (:predicates 
     (en ?caja ?posicion)
     (libre ?posicion)
  )
  (:action mover
    :parameters (?caja ?pos_actual ?pos_nueva)
    :precondition (and (en ?caja ?pos_actual) (libre ?pos_nueva))
    :effect (and (not (en ?caja ?pos_actual)) (en ?caja ?pos_nueva))
  )
)"""

STRIPS_PROBLEM = """(define (problem reordenar_cajas_problema)
  (:domain reordenar_cajas)
  (:init 
     (en caja1 pos1)
     (en caja2 pos2)
     (en caja3 pos3)
     (en caja4 pos4)
     (libre pos5)
  )
  (:goal (and 
     (en caja1 pos3)
     (en caja2 pos1)
     (en caja3 pos4)
     (en caja4 pos2)
  ))
)"""

# ===============================
# Interfaz Gráfica con Tkinter
# ===============================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trabajo Final - Inteligencia Artificial")
        self.geometry("900x700")
        self.resizable(False, False)
        
        # Variables globales para la parte de visión
        self.dataset_loaded = False
        self.train_data = None
        self.train_labels = None
        
        # Crear Notebook (pestañas)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)
        
        self.create_vision_tab()
        self.create_astar_tab()
        self.create_strips_tab()
    
    # ----------------------------
    # Pestaña de Visión Artificial
    # ----------------------------
    def create_vision_tab(self):
        self.vision_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.vision_frame, text="Visión Artificial")
        
        # Botón para cargar dataset
        self.btn_load_dataset = ttk.Button(self.vision_frame, text="Cargar Dataset",
                                           command=self.load_dataset_action)
        self.btn_load_dataset.pack(pady=10)
        
        # Etiqueta para mostrar carpeta del dataset cargado
        self.lbl_dataset = ttk.Label(self.vision_frame, text="Dataset: No cargado")
        self.lbl_dataset.pack(pady=5)
        
        # Botón para seleccionar imagen de prueba
        self.btn_select_image = ttk.Button(self.vision_frame, text="Seleccionar imagen de prueba",
                                           command=self.select_image_action)
        self.btn_select_image.pack(pady=10)
        
        # Etiqueta para mostrar imagen seleccionada
        self.canvas_img = tk.Canvas(self.vision_frame, width=250, height=250, bg="gray")
        self.canvas_img.pack(pady=10)
        
        # Etiqueta para mostrar resultados
        self.lbl_result_knn = ttk.Label(self.vision_frame, text="KNN: -")
        self.lbl_result_knn.pack(pady=5)
        self.lbl_result_km = ttk.Label(self.vision_frame, text="K-means: -")
        self.lbl_result_km.pack(pady=5)
    
    def load_dataset_action(self):
        folder = filedialog.askdirectory(title="Selecciona la carpeta del dataset")
        if folder:
            try:
                self.train_data, self.train_labels = load_dataset(folder)
                if len(self.train_data) == 0:
                    messagebox.showerror("Error", "No se encontraron imágenes en el dataset.")
                    return
                self.dataset_loaded = True
                self.lbl_dataset.config(text=f"Dataset: {folder}")
                messagebox.showinfo("Éxito", f"Cargadas {len(self.train_data)} imágenes.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def select_image_action(self):
        if not self.dataset_loaded:
            messagebox.showwarning("Atención", "Primero carga un dataset.")
            return
        
        file_path = filedialog.askopenfilename(title="Selecciona una imagen de prueba",
                                               filetypes=[("Imagenes", "*.jpg *.png *.jpeg")])
        if file_path:
            try:
                # Muestra la imagen en el canvas (redimensionada)
                img = Image.open(file_path)
                img.thumbnail((250, 250))
                self.photo = ImageTk.PhotoImage(img)
                self.canvas_img.create_image(125, 125, image=self.photo)
                
                # Extrae características de la imagen de prueba
                test_features = extract_features(file_path)
                
                # Clasificación con KNN
                pred_knn = knn_classify(test_features, self.train_data, self.train_labels, k=3)
                self.lbl_result_knn.config(text=f"KNN: {pred_knn}")
                
                # Clasificación con K-means
                centroids, clusters = kmeans(self.train_data, k=4)
                km_cluster = classify_kmeans(test_features, centroids)
                # Mapea cluster a etiqueta mayoritaria
                cluster_labels = []
                for j, points in clusters.items():
                    if points:
                        indices = [np.where(np.all(self.train_data == point, axis=1))[0][0] for point in points]
                        labels_in_cluster = [self.train_labels[i] for i in indices]
                        majority_label = Counter(labels_in_cluster).most_common(1)[0][0]
                        cluster_labels.append(majority_label)
                    else:
                        cluster_labels.append("Desconocido")
                pred_km = cluster_labels[km_cluster]
                self.lbl_result_km.config(text=f"K-means: {pred_km}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    # ----------------------------
    # Pestaña de A* Path Planning
    # ----------------------------
    def create_astar_tab(self):
        self.astar_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.astar_frame, text="A* Path Planning")
        
        # Botón para ejecutar A*
        self.btn_run_astar = ttk.Button(self.astar_frame, text="Ejecutar A*",
                                        command=self.run_astar_action)
        self.btn_run_astar.pack(pady=10)
        
        # Canvas para dibujar el grid y el camino
        self.canvas_astar = tk.Canvas(self.astar_frame, width=500, height=500, bg="white")
        self.canvas_astar.pack(pady=10)
        
        # Definimos un grid de 10x10 (0: libre, 1: obstáculo)
        self.astar_grid = [
            [0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,0,1,1,1,0],
            [0,1,0,0,0,0,0,0,1,0],
            [0,1,0,1,1,1,1,0,1,0],
            [0,0,0,0,0,0,1,0,0,0],
            [0,1,1,1,1,0,1,1,1,0],
            [0,1,0,0,0,0,0,0,1,0],
            [0,1,0,1,1,1,1,0,1,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,0]
        ]
        self.cell_size = 50  # Tamaño de cada celda en pixeles
        self.astar_start = (0, 0)
        self.astar_goal = (9, 9)
        self.draw_astar_grid()
    
    def draw_astar_grid(self, path=None):
        self.canvas_astar.delete("all")
        rows = len(self.astar_grid)
        cols = len(self.astar_grid[0])
        for i in range(rows):
            for j in range(cols):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                color = "white"
                if self.astar_grid[i][j] == 1:
                    color = "black"
                self.canvas_astar.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray")
        
        # Dibuja start y goal
        self.draw_cell(self.astar_start, "green")
        self.draw_cell(self.astar_goal, "red")
        
        # Dibuja el camino si existe
        if path:
            for cell in path:
                if cell != self.astar_start and cell != self.astar_goal:
                    self.draw_cell(cell, "yellow")
    
    def draw_cell(self, cell, color):
        i, j = cell
        x0 = j * self.cell_size
        y0 = i * self.cell_size
        x1 = x0 + self.cell_size
        y1 = y0 + self.cell_size
        self.canvas_astar.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray")
    
    def run_astar_action(self):
        path = a_star(self.astar_grid, self.astar_start, self.astar_goal)
        if path:
            self.draw_astar_grid(path)
            messagebox.showinfo("A*", f"Camino encontrado con {len(path)} pasos.")
        else:
            messagebox.showwarning("A*", "No se encontró camino.")
    
    # ----------------------------
    # Pestaña de STRIPS (PDDL)
    # ----------------------------
    def create_strips_tab(self):
        self.strips_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.strips_frame, text="STRIPS Planning")
        
        lbl_info = ttk.Label(self.strips_frame, text="Edita el dominio y problema STRIPS:")
        lbl_info.pack(pady=5)
        
        # Texto para dominio
        lbl_domain = ttk.Label(self.strips_frame, text="Dominio (PDDL):")
        lbl_domain.pack(anchor="w", padx=10)
        self.txt_domain = tk.Text(self.strips_frame, width=100, height=10)
        self.txt_domain.pack(padx=10, pady=5)
        self.txt_domain.insert("end", STRIPS_DOMAIN)
        
        # Texto para problema
        lbl_problem = ttk.Label(self.strips_frame, text="Problema (PDDL):")
        lbl_problem.pack(anchor="w", padx=10)
        self.txt_problem = tk.Text(self.strips_frame, width=100, height=10)
        self.txt_problem.pack(padx=10, pady=5)
        self.txt_problem.insert("end", STRIPS_PROBLEM)
        
        # Botón para guardar archivos
        self.btn_save_strips = ttk.Button(self.strips_frame, text="Guardar archivos PDDL",
                                          command=self.save_strips_files)
        self.btn_save_strips.pack(pady=10)
    
    def save_strips_files(self):
        # Guarda los archivos en la carpeta seleccionada
        folder = filedialog.askdirectory(title="Selecciona carpeta para guardar PDDL")
        if folder:
            domain_path = os.path.join(folder, "strips_domain.pddl")
            problem_path = os.path.join(folder, "strips_problema.pddl")
            try:
                with open(domain_path, "w") as f:
                    f.write(self.txt_domain.get("1.0", "end"))
                with open(problem_path, "w") as f:
                    f.write(self.txt_problem.get("1.0", "end"))
                messagebox.showinfo("STRIPS", f"Archivos guardados en {folder}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = App()
    app.mainloop()
