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
import sounddevice as sd  # Importação para captura de áudio
import threading
import queue
import time
from collections import deque  # Para implementar a média móvel

USE_3D_MODEL = True

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

    # Normalizar os vértices para que o modelo fique centrado e em escala similar
    vertices -= np.mean(vertices, axis=0)
    max_extent = np.max(np.abs(vertices))
    vertices /= max_extent
    return vertices

# Lista de caminhos para os modelos
model_paths = ['forma01.obj', 'forma02.obj', 'forma03.obj']

if USE_3D_MODEL:
    # Carregar todos os modelos e armazená-los em uma lista
    models = [load_3d_model(path) for path in model_paths]
else:
    # Se não estiver usando modelos 3D, usar esferas
    models = [generate_sphere(max_points) for _ in range(len(model_paths))]

# Garantir que todos os modelos tenham o mesmo número de pontos
num_points = min([model.shape[0] for model in models])
models = [model[:num_points] for model in models]

# Posição inicial é o primeiro modelo
current_model_index = 0
original_pos = models[current_model_index]
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
scale = 0.7
speed = 0.4
amplitude = 0.2

# Variáveis para transição entre modelos
transition_time = 5.0  # Duração da transição em segundos
time_since_last_transition = 0.0
transition_in_progress = False
transition_start_time = 0.0
next_model_index = (current_model_index + 1) % len(models)
transition_start_pos = original_pos.copy()
transition_end_pos = models[next_model_index]

# Classe base para fontes de entrada
class InputSource:
    def get_value(self):
        return 0.0

# Implementação da fonte de entrada do microfone
class MicrophoneInput(InputSource):
    def __init__(self, smoothing=10):
        self.q = queue.Queue()
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()
        self.value = 0.0
        self.smoothing = smoothing
        self.values = deque(maxlen=smoothing)  # Armazena os últimos N valores

    def audio_callback(self, indata, frames, time_info, status):
        volume_norm = np.linalg.norm(indata) / frames
        self.q.put(volume_norm)

    def get_value(self):
        # Atualiza o valor apenas se houver novos dados
        while not self.q.empty():
            new_value = self.q.get()
            self.values.append(new_value)
            # Calcula a média dos últimos N valores
            self.value = np.mean(self.values)
        return self.value

# Você pode adicionar outras fontes de entrada aqui
# Por exemplo, uma fonte de entrada que usa um sinal senoidal
class SineWaveInput(InputSource):
    def __init__(self, frequency=1.0):
        self.frequency = frequency
        self.start_time = time.time()

    def get_value(self):
        elapsed_time = time.time() - self.start_time
        return (np.sin(2 * np.pi * self.frequency * elapsed_time) + 1) / 2  # Normalizado entre 0 e 1

# Selecione a fonte de entrada desejada
input_source = MicrophoneInput(smoothing=300)  # Aumente o valor de 'smoothing' para mais suavidade
# input_source = SineWaveInput(frequency=0.5)

def update(event):
    global pos, time_counter, amplitude, scale, speed
    global time_since_last_transition, transition_in_progress, transition_start_time
    global current_model_index, next_model_index, original_pos
    global transition_start_pos, transition_end_pos

    time_counter += event.dt
    time_since_last_transition += event.dt

    # Verifica se é hora de iniciar uma nova transição
    if not transition_in_progress and time_since_last_transition >= 20.0:
        transition_in_progress = True
        transition_start_time = time_counter
        time_since_last_transition = 0.0
        # Define o próximo modelo
        next_model_index = (current_model_index + 1) % len(models)
        transition_start_pos = original_pos.copy()
        transition_end_pos = models[next_model_index]

    # Gerencia a transição
    if transition_in_progress:
        t = (time_counter - transition_start_time) / transition_time
        if t >= 1.0:
            t = 1.0
            transition_in_progress = False
            current_model_index = next_model_index
            original_pos = models[current_model_index]
        else:
            # Interpola entre as posições iniciais e finais
            original_pos = (1 - t) * transition_start_pos + t * transition_end_pos

    # Obtém o valor da fonte de entrada
    input_value = input_source.get_value()

    # Use o valor suavizado para modificar os parâmetros
    dynamic_speed = speed + input_value * 0.5  # Ajuste o multiplicador conforme necessário
    dynamic_amplitude = amplitude + input_value * 0.5

    # Calcula o deslocamento para todas as partículas de forma vetorizada
    nx = original_pos[:, 0] * scale + time_counter * dynamic_speed
    ny = original_pos[:, 1] * scale
    nz = original_pos[:, 2] * scale

    # Usando vetorização para melhorar a performance
    # Usando um mapa de ruído para todas as partículas
    noise_values_x = np.array([
        pnoise3(x, y, z) for x, y, z in zip(nx, ny, nz)
    ])
    noise_values_y = np.array([
        pnoise3(x, y + time_counter * dynamic_speed, z) for x, y, z in zip(nx, ny, nz)
    ])
    noise_values_z = np.array([
        pnoise3(x, y, z + time_counter * dynamic_speed) for x, y, z in zip(nx, ny, nz)
    ])

    displacement = np.vstack((noise_values_x, noise_values_y, noise_values_z)).T * dynamic_amplitude
    pos = original_pos + displacement

    displacement_magnitude = np.linalg.norm(displacement, axis=1)
    max_displacement_magnitude = dynamic_amplitude * np.sqrt(3)
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
