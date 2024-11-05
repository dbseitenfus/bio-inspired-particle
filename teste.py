# -*- coding: utf-8 -*-
import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
import trimesh
from noise import pnoise3
import matplotlib.pyplot as plt
import random

USE_3D_MODEL = True

# Configurações para os modelos 3D
model_paths = ["sphere"]
random.shuffle(model_paths)
max_points = 10000

# Configuração inicial
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()

# Função para gerar um ponto de impacto
def get_wave_origin():
    return np.array([0.0, 0.0, 0.0])  # Ponto central para as ondas

# Carrega o modelo 3D
def load_3d_model(file_path):
    if file_path == "sphere":
        vertices = generate_sphere(max_points)
    else:
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.Scene):
            combined = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            mesh = combined
        elif isinstance(mesh, trimesh.Trimesh):
            pass  # mesh já está carregada
        else:
            raise TypeError("O arquivo não contém uma malha compatível.")
        vertices, face_indices = trimesh.sample.sample_surface(mesh, max_points)
        vertices -= np.mean(vertices, axis=0)
        max_extent = np.max(np.abs(vertices))
        vertices /= max_extent
    return vertices

# Gera pontos em uma esfera
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

if USE_3D_MODEL:
    models = [load_3d_model(path) for path in model_paths]
else:
    models = [generate_sphere(max_points) for _ in range(len(model_paths))]

current_model_index = 0
original_pos = models[current_model_index]
pos = original_pos.copy()

scatter = visuals.Markers()
scatter.set_data(pos, edge_width=0, face_color=(0.5, 0.8, 1, 0.8), size=2)
scatter.set_gl_state('translucent', depth_test=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
view.add(scatter)
view.camera = 'turntable'

# Variáveis para animação de ondas
time_counter = 0.0
wave_speed = 1.3  # Velocidade de propagação da onda
wave_frequency = 3.0  # Frequência das ondas
wave_amplitude = 0.2  # Amplitude das ondas
wave_origin = get_wave_origin()  # Ponto de impacto

# Função de atualização para efeito de onda
def update(event):
    global pos, time_counter

    time_counter += event.dt

    # Calcula a distância radial de cada ponto ao ponto de impacto
    radial_distances = np.linalg.norm(original_pos - wave_origin, axis=1)

    # Calcula o deslocamento baseado em uma função seno para simular a onda
    wave_effect = wave_amplitude * np.sin(wave_frequency * (radial_distances - wave_speed * time_counter))

    # Adiciona o efeito de onda às partículas
    pos = original_pos + (original_pos - wave_origin) * wave_effect[:, np.newaxis]

    # Ajusta a cor das partículas com base na magnitude do deslocamento
    displacement_magnitude = np.abs(wave_effect)
    cmap = plt.get_cmap('plasma')
    colors = cmap(displacement_magnitude / np.max(displacement_magnitude))
    
    # Modifica o tamanho das partículas com base no deslocamento para destacar as ondas
    sizes = 3 + 10 * displacement_magnitude

    # Atualiza o scatter plot
    scatter.set_data(pos, edge_width=0, face_color=colors, size=sizes)
    canvas.update()

timer = vispy.app.Timer()
timer.connect(update)
timer.start(0.016)

if __name__ == '__main__':
    vispy.app.run()
