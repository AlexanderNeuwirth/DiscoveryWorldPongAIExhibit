import json
import paho.mqtt.client as mqtt
import numpy as np

class GameSubscriber:
    def emit_state(self, state, request_action=False):
        (puck_x, puck_y), left_y, right_y, score_left, score_right, level, frame = state

        self.client.publish("puck/position", payload=json.dumps({"x": puck_x, "y": puck_y}))
        self.client.publish("paddle1/position", payload=json.dumps({"position": left_y}))
        self.client.publish("paddle2/position", payload=json.dumps({"position": right_y}))
        self.client.publish("player1/score", payload=json.dumps({"score": score_left}))
        self.client.publish("player2/score", payload=json.dumps({"score": score_right}))
        self.client.publish("game/level", payload=json.dumps({"level": level}))
        if request_action:
            self.client.publish("game/frame", payload=json.dumps({"frame": frame}))

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("paddle1/action")
        client.subscribe("paddle2/action")
        client.subscribe("paddle2/frame")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = json.loads(msg.payload)
        if topic == "paddle1/action":
            self.paddle1_action = int(payload["action"])
        if topic == "paddle2/action":
            self.paddle2_action = int(payload["action"])
        if topic == "paddle2/frame":
            self.paddle2_frame = payload["frame"]

    def __init__(self):
        self.client = mqtt.Client()
        self.client.connect_async("localhost", port=1883, keepalive=60)
        self.client.on_connect = lambda client, userdata, flags, rc : self.on_connect(client, userdata, flags, rc)
        self.client.on_message = lambda client, userdata, msg : self.on_message(client, userdata, msg)
        self.client.loop_start()
        self.paddle1_action = 2 # "NONE"
        self.paddle1_prob = np.array([0, 1])
        self.paddle2_action = 2 # "NONE"
        self.paddle2_prob = np.array([0, 1])
        self.paddle2_frame = None
