# -*- coding: utf-8 -*-
# vispy: gallery 10
# Distributed under the (new) BSD License.

"""
Create and Animate a Point Cloud with Amoeba-like Movement
==========================================================

Demonstrates how to animate particles to simulate an amoeba or a living organism
by applying smooth, organic movements to the point cloud.
"""

import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
from noise import pnoise3  # Importa o ruído Perlin
import trimesh
import matplotlib.pyplot as plt  # Importa o pyplot para acessar get_cmap

USE_3D_MODEL = False

# Se o modelo tiver muitos vértices, você pode querer reduzir
max_points = 50000  # Ajuste conforme necessário

# Cria um canvas e adiciona uma visualização simples
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()

# Função para gerar pontos uniformemente distribuídos em uma esfera
def generate_sphere(num_points):
    phi = np.random.uniform(0, np.pi * 2, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    u = np.random.uniform(0, 1, num_points)

    theta = np.arccos(costheta)
    r = u ** (1/3)  # Distribuição uniforme no volume da esfera

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    pos = np.vstack((x, y, z)).T
    return pos

# Load a 3D model using trimesh
def load_3d_model(file_path):
    mesh = trimesh.load(file_path)
    
    if isinstance(mesh, trimesh.Scene):
        # Combina todas as malhas em uma única malha
        combined = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        vertices = combined.vertices
    elif isinstance(mesh, trimesh.Trimesh):
        vertices = mesh.vertices
    else:
        raise TypeError("O arquivo não contém uma malha compatível.")
    
    return vertices

# Path to your 3D model file (replace with your file path)
model_path = 'model.obj'

# Gera ou carrega as posições iniciais das partículas
if USE_3D_MODEL:
    # Load the model and get the vertices
    original_pos = load_3d_model(model_path)
else:
    original_pos = generate_sphere(max_points)

# Número de pontos
num_points = original_pos.shape[0]
print(f"Número de vértices: {num_points}")

# Copia as posições para a animação
pos = original_pos.copy()

# Cria o objeto scatter e preenche os dados
scatter = visuals.Markers()
scatter.set_data(pos, edge_width=0, face_color=(0.5, 0.8, 1, 0.8), size=2)

view.add(scatter)

view.camera = 'turntable'  # ou experimente 'arcball'

# Adiciona um eixo 3D colorido para orientação
axis = visuals.XYZAxis(parent=view.scene)

# Variáveis para a animação
time_counter = 0.0

def update(event):
    global pos, time_counter

    time_counter += event.dt

    # Parâmetros para a função de ruído
    scale = 1.4       # Escala espacial
    speed = 0.3       # Velocidade temporal
    amplitude = 0.3   # Amplitude máxima de deslocamento

    # Calcula o deslocamento usando ruído Perlin
    displacement = np.zeros_like(pos)

    for i in range(num_points):
        x, y, z = original_pos[i]
        nx = x * scale
        ny = y * scale
        nz = z * scale
        nt = time_counter * speed

        dx = pnoise3(nx + nt, ny, nz)
        dy = pnoise3(nx, ny + nt, nz)
        dz = pnoise3(nx, ny, nz + nt)

        displacement[i] = np.array([dx, dy, dz])

    displacement *= amplitude

    # Atualiza as posições
    pos = original_pos + displacement

    # Calcula a magnitude do deslocamento para mapear as cores
    displacement_magnitude = np.linalg.norm(displacement, axis=1)
    max_displacement_magnitude = amplitude * np.sqrt(3)
    normalized_magnitude = displacement_magnitude / max_displacement_magnitude
    normalized_magnitude = np.clip(normalized_magnitude, 0, 1)

    # Mapeia a magnitude normalizada para cores usando um colormap
    cmap = plt.get_cmap('viridis')
    colors = cmap(normalized_magnitude)

    # Atualiza os tamanhos das partículas
    sizes = 3 + normalized_magnitude * 3  # Tamanhos entre 2 e 5

    # Atualiza os dados do scatter
    scatter.set_data(pos, edge_width=0, face_color=colors, size=sizes)
    canvas.update()

timer = vispy.app.Timer()
timer.connect(update)
timer.start(0.016)  # Aproximadamente 60 quadros por segundo

if __name__ == '__main__':
    vispy.app.run()