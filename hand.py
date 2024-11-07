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
from collections import deque
import sounddevice as sd  # Importado para capturar áudio do microfone
import paho.mqtt.client as mqtt
import random
from trimesh.sample import sample_surface

import cv2
import mediapipe as mp  # Importado para detecção de mãos

import gc
gc.disable()


USE_3D_MODEL = True

# Configurações para os modelos 3D
model_paths = ["sphere", "corona.obj","corona.obj","corona.obj","Forma01.obj", "Forma02.obj", "Forma03.obj", "Forma04.obj", "Forma05.obj", "Forma06.obj", "Forma07.obj", "Forma08.obj", "Forma09.obj", "Forma10.obj", "Forma11.obj", "Forma12.obj", "Forma13.obj", "Forma14.obj", "Forma15.obj", "Forma16.obj", "Forma17.obj", "Forma18.obj", "Forma19.obj", "Forma20.obj", "Forma21.obj", "Forma22.obj", "Forma23.obj", "Forma24.obj", "Forma25.obj", "Forma26.obj", "Forma27.obj", "Forma28.obj", "Forma29.obj", "Forma30.obj", "Forma31.obj", "Forma32.obj", "Forma33.obj", "Forma34.obj", "Forma35.obj", "Forma36.obj", "Forma37.obj", "Forma38.obj", "Forma39.obj"]
random.shuffle(model_paths)
max_points = 20000  # Aumentado de 10000 para 50000

# Configurações de conexão MQTT
MQTT_IP = "34.27.98.205"
MQTT_PORT = 2494
MQTT_USER = "participants"
MQTT_PASSWORD = "prp1nterac"
MQTT_TOPICS = ["hiper/touch","hiper/touch2"]

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
        self.client.on_disconnect = self.on_disconnect
        self.ip = ip
        self.port = port

    def connect_and_loop(self):
        try:
            self.client.connect(self.ip, self.port, keepalive=120)
            # print("Conectado ao broker MQTT.")
            self.client.loop_forever()
        except Exception as e:
            print(f"Erro ao conectar ao broker MQTT: {e}")

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            # print("Desconexão inesperada. Tentando reconectar...")
            try:
                client.reconnect()
            except Exception as e:
                print(f"Erro ao tentar reconectar: {e}")


    def on_connect(self, client, userdata, flags, rc):
        print("Conectado ao broker MQTT com código de resultado: " + str(rc))
        for topic in self.topics:
            client.subscribe(topic)
            # print(f"Inscrito no tópico: {topic}")

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = float(msg.payload.decode())
            self.input_source.update_value(topic, payload)
            # print(f"Recebido do tópico {topic}: {payload}")
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
        # Amostrar pontos da superfície
        vertices, face_indices = trimesh.sample.sample_surface(mesh, max_points)
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
# num_points = min([model.shape[0] for model in models])
# models = [model[:num_points] for model in models]

# Posição inicial é o primeiro modelo
current_model_index = 0
original_pos = models[current_model_index]
pos = original_pos.copy()

# cmap = plt.get_cmap('plasma')  # Alterado de 'gnuplot' para 'plasma' para cores mais vibrantes

# Configuração do scatter plot com blending ajustado
scatter = visuals.Markers()
scatter.set_data(pos, edge_width=0, face_color=(0.5, 0.8, 1, 0.8), size=2)
scatter.set_gl_state('translucent', depth_test=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
view.add(scatter)
view.camera = 'turntable'

# Variáveis para a animação
time_counter = 0.0
scale = 0.0
speed = 0.0
amplitude = 0.0

scale_goal = 1
speed_goal = 0.4
amplitude_goal = 1.5

from matplotlib import colormaps

# Precarregar o colormap
color_style_change_interval = 60.0  # Intervalo em segundos (4 minutos)
time_since_last_color_change = 0.0
current_color_style_index = 0
color_styles = ['plasma', 'viridis', 'magma', 'cividis', 'hsv', 'tab10', 'tab20', 'prism', 'nipy_spectral', 'gist_stern']  # Lista de colormaps
random.shuffle(color_styles)
cmap = plt.get_cmap(color_styles[current_color_style_index])  # Colormap inicial
# Variáveis para controlar a transição de cor
color_transition_duration = 10.0  # Duração da transição em segundos
color_transition_in_progress = False
color_transition_start_time = 0.0
previous_cmap = cmap  # Armazena o colormap anterior


# Variáveis para transição entre modelos
transition_time = 5.0  # Duração da transição em segundos
time_since_last_transition = 0.0
transition_in_progress = False
transition_start_time = 0.0
next_model_index = (current_model_index + 1) % len(models)
transition_start_pos = original_pos.copy()
transition_end_pos = models[next_model_index]

# Instanciar a fonte de entrada MQTT
input_source = MqttInput(None)

# Conectar ao servidor MQTT
client = connect_mqtt_server(input_source)
input_source.client = client

# Variáveis para o volume do microfone
mic_volume = 0.0
stream = None  # Variável global para manter a referência ao stream

def audio_callback(indata, frames, time, status):
    global mic_volume
    # Calcula o volume RMS (Root Mean Square)
    volume_norm = np.linalg.norm(indata) * 10
    mic_volume = volume_norm

def start_audio_stream():
    global stream
    stream = sd.InputStream(callback=audio_callback)
    stream.start()

# Inicia o stream de áudio na thread principal
start_audio_stream()

# Função para publicar os dados no servidor MQTT
def publish_data(client, topic, data):
    try:
        client.client.publish(topic, data)
        # print(f"Dados enviados para {topic}: {data}")
    except Exception as e:
        print(f"Erro ao enviar dados para {topic}: {e}")

# Função para enviar os dados a cada 1 segundo
def send_data_periodically(event):
    # global scale, speed, amplitude, client
    publish_data(client, "hiper/labinter/scale", scale)
    publish_data(client, "hiper/labinter/speed", speed)
    publish_data(client, "hiper/labinter/amplitude", amplitude)

data_timer = vispy.app.Timer(interval=0.2, start=True, connect=send_data_periodically)

def update_goals(event):
    global scale_goal, speed_goal, amplitude_goal
    # Adicionar aleatoriedade dentro de um intervalo menor em torno dos valores atuais
    scale_goal = random.uniform(max(0.5, scale_goal - 0.5), min(2.0, scale_goal + 0.5))
    # speed_goal = random.uniform(max(0.1, speed_goal - 0.3), min(1.0, speed_goal + 0.3))
    amplitude_goal = random.uniform(max(0.5, amplitude_goal - 1.0), min(3.0, amplitude_goal + 1.0))
    gc.collect()

# Temporizador para atualizar os goals a cada 40 segundos
goal_timer = vispy.app.Timer(interval=100.0, start=True, connect=update_goals)

# Variáveis para a detecção de mãos
hand_detected = False
num_hands = 0
hand_lock = threading.Lock()
hand_landmarks_list = []  # Armazena os landmarks das mãos

# Função para detecção de mãos em outra thread
def hand_detection_thread():
    global hand_detected, num_hands, hand_landmarks_list
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)  # Abre a câmera padrão

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Converte a imagem para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Processa a imagem para detecção de mãos
        results = hands.process(frame_rgb)

        with hand_lock:
            if results.multi_hand_landmarks:
                hand_detected = True
                num_hands = len(results.multi_hand_landmarks)
                hand_landmarks_list = []
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extrair as coordenadas dos landmarks
                    hand_landmarks_list.append([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            else:
                hand_detected = False
                num_hands = 0
                hand_landmarks_list = []

    cap.release()
    cv2.destroyAllWindows()

# Inicia a thread de detecção de mãos
hand_thread = threading.Thread(target=hand_detection_thread)
hand_thread.daemon = True
hand_thread.start()

# Criar um visual para os landmarks das mãos
hand_scatter = visuals.Markers()
hand_scatter.set_data(np.array([[0, 0, 0]]), face_color=(1, 1, 1, 1), size=5)
view.add(hand_scatter)

def update(event):
    global pos, time_counter, amplitude, scale, speed, scale_goal, speed_goal, amplitude_goal
    global time_since_last_transition, transition_in_progress, transition_start_time
    global current_model_index, next_model_index, original_pos
    global transition_start_pos, transition_end_pos
    global time_since_last_color_change, color_style_change_interval, current_color_style_index, color_styles, cmap
    global color_transition_duration, color_transition_in_progress, color_transition_start_time, previous_cmap
    global hand_landmarks_list

    time_counter += event.dt
    time_since_last_transition += event.dt
    time_since_last_color_change += event.dt  # Atualiza o temporizador de cores

     # Verifica se é hora de iniciar uma nova transição de cor
    if time_since_last_color_change >= color_style_change_interval:
        time_since_last_color_change = 0.0  # Reinicia o temporizador
        # Inicia a transição de cor
        color_transition_in_progress = True
        color_transition_start_time = time_counter
        previous_cmap = cmap
        current_color_style_index = (current_color_style_index + 1) % len(color_styles)
        cmap = plt.get_cmap(color_styles[current_color_style_index])

    # Verifica se é hora de iniciar uma nova transição
    if not transition_in_progress and time_since_last_transition >= 40.0:
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

    # # Obtém os valores dos sensores
    # touch = input_source.get_value("hiper/touch")
    # touch2 = input_source.get_value("hiper/touch2")
    # luminosity = input_source.get_value("sensor/luminosity")
    # humidity = input_source.get_value("sensor/humidity")

    # # Definir os valores mínimo e máximo esperados para normalização
    # temp_min, temp_max = 0, 1       # Exemplo para temperatura (°C)
    # ph_min, ph_max = 0, 1            # Exemplo para pH
    # lum_min, lum_max = 0, 1000       # Exemplo para luminosidade
    # hum_min, hum_max = 0, 100        # Exemplo para umidade (%)

    # # Normalizar os valores dos sensores
    # normalized_temp = (touch - temp_min) / (temp_max - temp_min)
    # normalized_temp = np.clip(normalized_temp, 0, 1)

    # normalized_ph = (touch2 - ph_min) / (ph_max - ph_min)
    # normalized_ph = np.clip(normalized_ph, 0, 1)

    # normalized_lum = (luminosity - lum_min) / (lum_max - lum_min)
    # normalized_lum = np.clip(normalized_lum, 0, 1)

    # normalized_hum = (humidity - hum_min) / (hum_max - hum_min)
    # normalized_hum = np.clip(normalized_hum, 0, 1)

    # Use os valores normalizados para modificar os parâmetros
    # speed = normalized_ph * 0.2         # Reduzido de 0.5 para 0.2
    # amplitude = normalized_temp * 0.2  # Reduzido de 0.5 para 0.2

    # scale = scale + scale_goal * 0.01
    # speed = speed + speed_goal * 0.01
    # amplitude = amplitude + amplitude_goal * 0.01

    touch = input_source.get_value("hiper/touch")
    touch2 = input_source.get_value("hiper/touch2")

    rgb = input_source.get_value("hiper/touch")

    # if random.choice([0,1]) == 1:
    #     scale_goal = scale_goal + touch * 0.001
    #     amplitude_goal = amplitude_goal + touch2 * 0.001
    # else:
    #     scale_goal = scale_goal - touch * 0.001
    #     amplitude_goal = amplitude_goal - touch2 * 0.001
    #     speed_goal = speed_goal - touch2 * 0.001


    # # Normalizar o volume do microfone
    normalized_mic_volume = np.clip(mic_volume / 10.0, 0.0, 0.5)
    # # print(normalized_mic_volume)

    # # Atualizar a velocidade com base no volume do microfone
    # amplitude_goal = normalized_mic_volume * 10
    amplitude_goal = normalized_mic_volume * 5

    # print(amplitude_goal)

    # speed_goal = normalized_mic_volume

    # print(speed_goal)

    # Ajustar scale, speed e amplitude suavemente
    scale += (scale_goal - scale) * 0.01
    speed += (speed_goal - speed) * 0.001
    amplitude += (amplitude_goal - amplitude) * 0.07

    # Calcula o deslocamento para todas as partículas de forma vetorizada
    nx = original_pos[:, 0] * scale + time_counter * speed
    ny = original_pos[:, 1] * scale + time_counter * speed
    nz = original_pos[:, 2] * scale + time_counter * speed

    # Usando vetorização para melhorar a performance
    noise_values_x = np.array([
        pnoise3(x, y, z) for x, y, z in zip(nx, ny, nz)
    ])
    noise_values_y = np.array([
        pnoise3(x, y + time_counter * speed, z) for x, y, z in zip(nx, ny, nz)
    ])
    noise_values_z = np.array([
        pnoise3(x, y, z + time_counter * speed) for x, y, z in zip(nx, ny, nz)
    ])

    displacement = np.vstack((noise_values_x, noise_values_y, noise_values_z)).T * amplitude
    pos = original_pos + displacement

    displacement_magnitude = np.linalg.norm(displacement, axis=1)
    max_displacement_magnitude = amplitude * np.sqrt(3)
    normalized_magnitude = np.clip(displacement_magnitude / max_displacement_magnitude, 0, 1)

    # Atualizar os landmarks das mãos
    with hand_lock:
        if hand_landmarks_list:
            # Concatenar os landmarks de todas as mãos
            landmarks = np.array(hand_landmarks_list).reshape(-1, 3)
            # Converter as coordenadas dos landmarks para o espaço 3D do VisPy
            # Os valores de x e y estão em relação à largura e altura da imagem (0 a 1)
            # Precisamos ajustar para o sistema de coordenadas do VisPy
            landmarks_vispy = landmarks.copy()
            # Inverter o eixo y (já que as coordenadas da imagem começam no topo)
            landmarks_vispy[:, 1] = 1 - landmarks_vispy[:, 1]
            # Centralizar e escalar os landmarks
            landmarks_vispy = (landmarks_vispy - 0.5) * 2  # Ajuste conforme necessário
            # Opcional: Aplicar uma profundidade (eixo z) fixa ou baseada nos dados
            # Aqui, multiplicamos por -1 para inverter a profundidade (dependendo da sua cena)
            landmarks_vispy[:, 2] *= -1
            hand_scatter.set_data(landmarks_vispy, face_color=(1, 1, 1, 1), size=10)
        else:
            # Se não houver landmarks, limpar os dados
            hand_scatter.set_data(np.empty((0, 3)))

    # Se a transição de cor está em progresso, faça a interpolação gradual
    if color_transition_in_progress:
        t_color = (time_counter - color_transition_start_time) / color_transition_duration
        t_color = np.clip(t_color, 0.0, 1.0)  # Limita t_color a 1.0 no máximo
        if t_color >= 1.0:
            color_transition_in_progress = False  # Finaliza a transição
        
        # Interpolação entre os colormaps antigo e novo
        colors_old = previous_cmap(normalized_magnitude)
        colors_new = cmap(normalized_magnitude)
        colors = (1 - t_color) * colors_old + t_color * colors_new
    else:
        # Se não há transição, use o colormap atual diretamente
        colors = cmap(normalized_magnitude)

    # Mapeia a magnitude do deslocamento para a cor
    # colors = cmap(normalized_magnitude)

    # Ajustar a transparência com base na magnitude do deslocamento
    alpha = 0.5 + 0.5 * normalized_magnitude  # Varia entre 0.5 e 1.0
    colors[:, 3] = alpha

    # Modificar o tamanho das partículas
    sizes = 5 + normalized_magnitude * 5  # Partículas maiores para deslocamentos maiores

    scatter.set_data(pos, edge_width=0, face_color=colors, size=sizes)
    canvas.update()

# (A função update_sensor_texts e a conexão com o evento draw permanecem iguais)

# Eventos de teclado para ajustar parâmetros manualmente (permanece igual)

timer = vispy.app.Timer()
timer.connect(update)
timer.start(0.016)

if __name__ == '__main__':
    vispy.app.run()
