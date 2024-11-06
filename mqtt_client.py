from paho.mqtt.client import Client


'''Este script OBRIGATORIAMENTE necessita do módulo "Paho MQTT",
  que pode ser instalado com o seguinte comando (no CMD ou no Terminal):
    => pip3 install paho-mqtt

  Código por:  João V. Coelho
  https://github.com/joaovcoelho

Desenvolvido em Python3'''

class MqttClient:
	def __init__(self, topics, clientid, ip, port, user, password):
		self.topics = topics
		self.clientid = clientid
		self.ip = ip
		self.port = port 
		self.user = user
		self.password = password

	# Apresenta o início da sessão e faz a subscrição em tópicos
	def on_connect(self, client, userdata, flags, rc):
		print(f"Connected {client._client_id}")
		print(f"Connection result code: [{rc}]")
		for topic in self.topics:
			client.subscribe(topic=topic, qos= 2)

	# Continuamente escuta e apresenta o recebimento dos dados	
	def on_message(self, client, userdata, message):
		# print(f"\033[1;42m{str(message.topic)}\033[m", message.payload)
		message.payload = message.payload.decode("utf-8") # Decodifica de binário para Unicode UTF-8
			
		if  message.topic in self.topics:
			print("="*15)
			print(f"Topic: {message.topic}")
			print(f"Payload: {message.payload}")
			print(f"QoS: {message.qos}")
			
	def on_disconnect(self, client, userdata, rc):
		if rc != 0:
			print("Desconexão inesperada. Tentando reconectar...")
			try:
				client.reconnect()
			except Exception as e:
				print(f"Erro ao tentar reconectar: {e}")

	# inicia a conexão em si, bem como a mantém ativa	
	def connect(self):
		client = Client(client_id= self.clientid, clean_session= False) # chama o módulo
		client.on_connect = self.on_connect # instancia a função de conexão
		client.on_message = self.on_message # instancia a função de recebimento de dados
		client.on_disconnect = self.on_disconnect
		
		client.username_pw_set(self.user, self.password) # Insere as credenciais para conexão
		client.connect(host=self.ip, port=self.port, keepalive=120) # inicia a conexão, de fato
		
		client.loop_forever() # mantém a conexão ativa
