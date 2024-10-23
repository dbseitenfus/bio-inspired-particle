# -*- coding: utf-8 -*-
# vispy: gallery 10
# Distributed under the (new) BSD License.

import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
from noise import pnoise3
import trimesh
import matplotlib.pyplot as plt

USE_3D_MODEL = False

# Se o modelo tiver muitos vértices, você pode querer reduzir
max_points = 10000

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()

# Função para gerar pontos uniformemente distribuídos em uma esfera
def generate_sphere(num_points):
    phi = np.random.uniform(0, np.pi * 2, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    u = np.random.uniform(0, 1, num_points)

    theta = np.arccos(costheta)
    r = u ** (1 / 3)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    pos = np.vstack((x, y, z)).T
    return pos

# Load a 3D model using trimesh
def load_3d_model(file_path):
    mesh = trimesh.load(file_path)

    if isinstance(mesh, trimesh.Scene):
        combined = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        vertices = combined.vertices
    elif isinstance(mesh, trimesh.Trimesh):
        vertices = mesh.vertices
    else:
        raise TypeError("O arquivo não contém uma malha compatível.")
    
    return vertices

model_path = 'star.obj'

if USE_3D_MODEL:
    original_pos = load_3d_model(model_path)
else:
    original_pos = generate_sphere(max_points)

num_points = original_pos.shape[0]
pos = original_pos.copy()

scatter = visuals.Markers()
scatter.set_data(pos, edge_width=0, face_color=(0.5, 0.8, 1, 0.8), size=2)
view.add(scatter)
view.camera = 'turntable'

axis = visuals.XYZAxis(parent=view.scene)

# Precarregar o colormap para evitar recomputação
cmap = plt.get_cmap('gnuplot')

# Variáveis para a animação
time_counter = 0.0
scale = 1
speed = 0.1
amplitude = 0.4

def update(event):
    global pos, time_counter, amplitude, scale, speed

    # speed += 0.01
    amplitude += 0.001

    time_counter += event.dt

    # Calcula o deslocamento para todas as partículas de forma vetorizada
    nx = original_pos[:, 0] * scale + time_counter * speed
    ny = original_pos[:, 1] * scale
    nz = original_pos[:, 2] * scale

    dx = np.array([pnoise3(x, y, z) for x, y, z in zip(nx, ny, nz)])
    dy = np.array([pnoise3(x, y + time_counter * speed, z) for x, y, z in zip(nx, ny, nz)])
    dz = np.array([pnoise3(x, y, z + time_counter * speed) for x, y, z in zip(nx, ny, nz)])

    displacement = np.vstack((dx, dy, dz)).T * amplitude
    pos = original_pos + displacement

    displacement_magnitude = np.linalg.norm(displacement, axis=1)
    max_displacement_magnitude = amplitude * np.sqrt(3)
    normalized_magnitude = np.clip(displacement_magnitude / max_displacement_magnitude, 0, 1)

    # Atualizar cores e tamanhos
    colors = cmap(normalized_magnitude)
    sizes = 3 + normalized_magnitude * 3

    scatter.set_data(pos, edge_width=0, face_color=colors, size=sizes)
    canvas.update()

def add_speed(value):
    global speed
    speed += value

def add_scale(value):
    global scale
    scale += value

def add_amplitude(value):
    global amplitude
    amplitude += value

@canvas.events.key_press.connect
def on_key_press(event):
    if event.key == 'a':
        add_speed(0.01)
    if event.key == 'b':
        add_scale(0.01)
    if event.key == 'c':
        add_amplitude(0.01)

timer = vispy.app.Timer()
timer.connect(update)
timer.start(0.016)

if __name__ == '__main__':
    vispy.app.run()
