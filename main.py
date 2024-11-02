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
import threading
import queue
from collections import deque
import time

import paho.mqtt.client as mqtt

from hands_pose import *

USE_3D_MODEL = True

hand_tracker = HandTracker()
# hand_tracker.run()

# Configurações para os modelos 3D
model_paths = ['forma01.obj', 'forma02.obj', 'forma03.obj']
max_points = 10000

# Configurações de conexão MQTT
MQTT_IP = "34.27.98.205"
MQTT_PORT = 2494
MQTT_USER = "participants"
MQTT_PASSWORD = "prp1nterac"
MQTT_TOPICS = ["sensor/temperature", "sensor/ph", "sensor/luminosity", "sensor/humidity"]

# Classe base para fontes de entrada
class InputSource:
    def get_value(self, topic):
        return 0.0

# Implementação da fonte de entrada MQTT
class MqttInput(InputSource):
    def __init__(self, client, smoothing=10):
        self.client = client
        self.values = {}
        self.lock = threading.Lock()
        self.smoothing = smoothing
        self.history = {}  # Armazena histórico dos valores para suavização

    def get_value(self, topic):
        with self.lock:
            if topic in self.history and len(self.history[topic]) > 0:
                return np.mean(self.history[topic])
            else:
                return 0.0

    def update_value(self, topic, value):
        with self.lock:
            if topic not in self.history:
                self.history[topic] = deque(maxlen=self.smoothing)
            self.history[topic].append(value)
            self.values[topic] = np.mean(self.history[topic])

# Implementação do cliente MQTT
class CustomMqttClient:
    def __init__(self, topics, input_source, clientid, ip, port, user, password):
        self.topics = topics
        self.input_source = input_source
        self.client = mqtt.Client(client_id=clientid)
        self.client.username_pw_set(user, password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.ip = ip
        self.port = port

    def connect_and_loop(self):
        try:
            self.client.connect(self.ip, self.port, keepalive=60)
            print("Conectado ao broker MQTT.")
            self.client.loop_forever()
        except Exception as e:
            print(f"Erro ao conectar ao broker MQTT: {e}")

    def on_connect(self, client, userdata, flags, rc):
        print("Conectado ao broker MQTT com código de resultado: " + str(rc))
        for topic in self.topics:
            client.subscribe(topic)
            print(f"Inscrito no tópico: {topic}")

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = float(msg.payload.decode())
            self.input_source.update_value(topic, payload)
            print(f"Recebido do tópico {topic}: {payload}")
        except Exception as e:
            print(f"Erro ao processar a mensagem do tópico {msg.topic}: {e}")

# Função para conectar ao servidor MQTT
def connect_mqtt_server(input_source):
    client = CustomMqttClient(
        MQTT_TOPICS,
        input_source,
        clientid="test",
        ip=MQTT_IP,
        port=MQTT_PORT,
        user=MQTT_USER,
        password=MQTT_PASSWORD
    )
    # Executa o loop do MQTT em um thread separado
    mqtt_thread = threading.Thread(target=client.connect_and_loop)
    mqtt_thread.daemon = True  # Permite que o thread seja encerrado ao fechar o programa
    mqtt_thread.start()
    return client

# Carrega o modelo 3D
def load_3d_model(file_path):
    mesh = trimesh.load(file_path)

    if isinstance(mesh, trimesh.Scene):
        combined = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        vertices = combined.vertices
    elif isinstance(mesh, trimesh.Trimesh):
        vertices = mesh.vertices
    else:
        raise TypeError("O arquivo não contém uma malha compatível.")

    # Normalizar os vértices
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

# Configuração inicial
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()

if USE_3D_MODEL:
    # Carregar todos os modelos
    models = [load_3d_model(path) for path in model_paths]
else:
    models = [generate_sphere(max_points) for _ in range(len(model_paths))]

# Garantir que todos os modelos tenham o mesmo número de pontos
num_points = min([model.shape[0] for model in models])
models = [model[:num_points] for model in models]

# Posição inicial é o primeiro modelo
current_model_index = 0
original_pos = models[current_model_index]
pos = original_pos.copy()

# Configuração do scatter plot
scatter = visuals.Markers()
scatter.set_data(pos, edge_width=0, face_color=(0.5, 0.8, 1, 0.8), size=2)
view.add(scatter)
view.camera = 'turntable'

axis = visuals.XYZAxis(parent=view.scene)

# Precarregar o colormap
cmap = plt.get_cmap('gnuplot')  # Usando o colormap original

# Variáveis para a animação
time_counter = 0.0
scale = 0.7
speed = 0.4
amplitude = 0.8

# Variáveis para transição entre modelos
transition_time = 5.0  # Duração da transição em segundos
time_since_last_transition = 0.0
transition_in_progress = False
transition_start_time = 0.0
next_model_index = (current_model_index + 1) % len(models)
transition_start_pos = original_pos.copy()
transition_end_pos = models[next_model_index]

# Instanciar a fonte de entrada MQTT
input_source = MqttInput(None)  # O cliente será atribuído após a conexão

# Conectar ao servidor MQTT
client = connect_mqtt_server(input_source)
input_source.client = client  # Atribuir o cliente à fonte de entrada

# Configuração do scatter plot para as partículas das mãos
hand_scatter = visuals.Markers()
view.add(hand_scatter)  # Adiciona o scatter das mãos à visualização

def update(event):
    global pos, time_counter, amplitude, scale, speed
    global time_since_last_transition, transition_in_progress, transition_start_time
    global current_model_index, next_model_index, original_pos
    global transition_start_pos, transition_end_pos


    time_counter += event.dt
    time_since_last_transition += event.dt

    # Verifica se é hora de iniciar uma nova transição
    if not transition_in_progress and time_since_last_transition >= 10.0:
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

    # Obtém os valores dos sensores
    temperature = input_source.get_value("sensor/temperature")
    ph = input_source.get_value("sensor/ph")
    luminosity = input_source.get_value("sensor/luminosity")
    humidity = input_source.get_value("sensor/humidity")

    # Definir os valores mínimo e máximo esperados para normalização
    temp_min, temp_max = 0, 40       # Exemplo para temperatura (°C)
    ph_min, ph_max = 0, 14           # Exemplo para pH
    lum_min, lum_max = 0, 1000       # Exemplo para luminosidade
    hum_min, hum_max = 0, 100        # Exemplo para umidade (%)

    # Normalizar os valores dos sensores
    normalized_temp = (temperature - temp_min) / (temp_max - temp_min)
    normalized_temp = np.clip(normalized_temp, 0, 1)

    normalized_ph = (ph - ph_min) / (ph_max - ph_min)
    normalized_ph = np.clip(normalized_ph, 0, 1)

    normalized_lum = (luminosity - lum_min) / (lum_max - lum_min)
    normalized_lum = np.clip(normalized_lum, 0, 1)

    normalized_hum = (humidity - hum_min) / (hum_max - hum_min)
    normalized_hum = np.clip(normalized_hum, 0, 1)

    # Use os valores normalizados para modificar os parâmetros
    dynamic_speed = speed + normalized_ph * 0.5         # pH influencia a velocidade
    dynamic_amplitude = amplitude + normalized_temp * 0.5  # Temperatura influencia a amplitude

    # Calcula o deslocamento para todas as partículas de forma vetorizada
    nx = original_pos[:, 0] * scale + time_counter * dynamic_speed
    ny = original_pos[:, 1] * scale
    nz = original_pos[:, 2] * scale

    # Usando vetorização para melhorar a performance
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

    # Mapeia a magnitude do deslocamento para a cor
    # colors = cmap(normalized_magnitude)

    # # Ajustar a transparência com base na luminosidade
    # alpha = np.clip(normalized_lum, 0.1, 1.0)
    # colors[:, 3] = alpha  # Atualiza o canal alfa das cores

    # # Modificar o tamanho das partículas com base na umidade
    # sizes = 3 + normalized_hum * 5  # Partículas maiores para umidade alta

    # Atualizar cores e tamanhos
    colors = cmap(normalized_magnitude)
    sizes = 3 + normalized_magnitude * 3

    scatter.set_data(pos, edge_width=0, face_color=colors, size=sizes)

    # Captura e renderiza as partículas das mãos
    hands_points = hand_tracker.get_hand_landmarks()
    if hands_points:
        hand_points_array = np.array(hands_points)

        # Ajusta os eixos X e Y para centralizar e manter as partículas alinhadas
        hand_points_array[:, 0] = (hand_points_array[:, 0] - 0.5) * 2  # Centraliza no eixo X
        hand_points_array[:, 1] = (0.5 - hand_points_array[:, 1]) * 2  # Inverte e centraliza o eixo Y
        
        # Define o eixo Z para um valor constante para dar a impressão de um plano 2D
        hand_points_array[:, 2] = -0.5  # Define Z para um plano fixo

        # Renderiza as partículas das mãos com um tamanho maior
        hand_scatter.set_data(
            hand_points_array, edge_width=0, face_color=(1, 0.5, 0.5, 0.9), size=10
        )  # Cor e tamanho para as partículas das mãos



    canvas.update()

# Opcional: Exibir valores dos sensores na tela
from vispy.scene.visuals import Text

temperature_text = Text('', color='white', font_size=12, parent=view.scene, pos=(10, 10))
ph_text = Text('', color='white', font_size=12, parent=view.scene, pos=(10, 30))
luminosity_text = Text('', color='white', font_size=12, parent=view.scene, pos=(10, 50))
humidity_text = Text('', color='white', font_size=12, parent=view.scene, pos=(10, 70))

def update_sensor_texts():
    temperature = input_source.get_value("sensor/temperature")
    ph = input_source.get_value("sensor/ph")
    luminosity = input_source.get_value("sensor/luminosity")
    humidity = input_source.get_value("sensor/humidity")

    temperature_text.text = f'Temperatura: {temperature:.2f}°C'
    ph_text.text = f'pH: {ph:.2f}'
    luminosity_text.text = f'Luminosidade: {luminosity:.2f} lx'
    humidity_text.text = f'Umidade: {humidity:.2f}%'

@canvas.events.draw.connect
def on_draw(event):
    update_sensor_texts()

# Eventos de teclado para ajustar parâmetros manualmente
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
