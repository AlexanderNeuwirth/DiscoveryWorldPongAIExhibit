from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import base64
from PIL import Image
import cv2
import numpy as np
from imageio import imread
import io
import time
import matplotlib.pyplot as plt
from random import choice


class DinoEnv:
    JUMP = 'JUMP'
    DUCK = 'DUCK'
    RUN = 'RUN'
    URL = "chrome://dino"
    ACTIONS = [JUMP, DUCK, RUN]
    SCREEN_SIZE = (150, 600)

    @staticmethod
    def preprocess_screen(screen):
        gray = cv2.cvtColor(screen, cv2.COLOR_RGBA2GRAY)
        downsample = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4))
        rescale = downsample.astype(np.float32) / 255.0
        return rescale

    def jump(self):
        if self.ducking:
            self.duck_stop_action.perform()
            self.ducking = False
        self.jump_start_action.perform()
        self.jump_stop_action.perform()

    def run(self):
        if self.ducking:
            self.duck_stop_action.perform()
            self.ducking = False

    def duck(self):
        if not self.ducking:
            self.duck_start_action.perform()
            self.ducking = True

    def get_score(self):
        score = int(self.driver.execute_script('return Runner.instance_.distanceRan/40;'))
        return score

    def get_dead(self):
        dead = self.driver.execute_script('return Runner.instance_.crashed;')
        return dead

    def get_screen(self):
        canvas_base64 = self.driver.execute_script("return arguments[0].toDataURL('image/png').substring(22);", self.canvas)
        return imread(io.BytesIO(base64.b64decode(canvas_base64)))

    def __init__(self):
        self.last_score = 0
        self.ducking = False
        self.driver = webdriver.Chrome()
        self.driver.get(DinoEnv.URL)
        self.jump_start_action = ActionChains(self.driver).key_down(Keys.ARROW_UP)
        self.jump_stop_action = ActionChains(self.driver).key_up(Keys.ARROW_UP)
        self.duck_start_action = ActionChains(self.driver).key_down(Keys.ARROW_DOWN)
        self.duck_stop_action = ActionChains(self.driver).key_up(Keys.ARROW_DOWN)
        self.canvas = None
        self.last_state = None

    def reset(self):
        self.last_score = 0
        self.ducking = False
        self.driver.refresh()
        self.canvas = self.driver.find_element_by_class_name("runner-canvas")
        self.driver.maximize_window()
        time.sleep(1)

        state = self.get_screen()
        self.last_state = state
        self.jump()
        return state

    def render(self, delay=0):
        if self.last_state is not None:
            cv2.imshow("Dino", self.last_state)
            cv2.waitKey(delay)

    def step(self, action):
        state = self.get_screen()
        if action == DinoEnv.JUMP:
            self.jump()
        if action == DinoEnv.RUN:
            self.run()
        if action == DinoEnv.DUCK:
            self.duck()
        reward = 0
        done = False
        if self.get_dead():
            reward = -1
            done = True
        self.last_state = state
        return state, reward, done
