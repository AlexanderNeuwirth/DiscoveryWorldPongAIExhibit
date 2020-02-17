from random import randint
from ai import DQN
from random import random, choice
import numpy as np
from pong import Pong
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from utils import save_video, write, plot_loss


class HumanPlayer:
    def __init__(self, up, down):
        self.up = up
        self.down = down

    def move(self, state):
        return Pong.read_key(self.up, self.down)


class RandomPlayer:
    def move(self, state):
        return Pong.random_action()


class BotPlayer:
    def __init__(self, env=None, left=False, right=False, always_follow=False):
        self.left = left
        self.right = right
        self.always_follow = always_follow
        if env is not None:
            self.paddle, self.ball = env.get_bot_data(left=left, right=right)

    def attach_env(self, env):
        self.paddle, self.ball = env.get_bot_data(left=self.left, right=self.right)

    def act(self, state):
        return self.move(), 1

    # Takes state to preserve interface
    def move(self):
        if self.always_follow:
            if self.ball.y > self.paddle.y:
                return 1
            elif self.ball.y < self.paddle.y:
                return 0
            else:
                return 0 if randint(0, 1) == 1 else 1
        if self.left and not self.ball.right or self.right and self.ball.right:
            if self.ball.y > self.paddle.y:
                return 1
            elif self.ball.y < self.paddle.y:
                return 0
            else:
                return 0 if randint(0, 1) == 1 else 1
        else:
            return 0 if randint(0, 1) == 1 else 1


class PGAgent:
    def __init__(self, state_size, action_size, name="PGAgent", learning_rate=0.001):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = learning_rate
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(200, activation='relu', init='he_uniform', input_shape=(self.state_size,)))
        model.add(Dense(100, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        prob = self.model.predict(state, batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards.astype(np.float32)

    def train(self, states, actions, probs, rewards):
        gradients = []
        for i in range(len(actions)):
            action = actions[i]
            prob = probs[i]
            y = np.zeros([self.action_size])
            y[action] = 1
            gradients.append(np.array(y).astype('float32') - prob)

        gradients = np.vstack(gradients)
        rewards = np.vstack(rewards)
        rewards = self.discount_rewards(rewards)
        gradients *= rewards

        X = np.squeeze(np.vstack([states]))
        Y = probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        result = self.model.train_on_batch(X, Y)
        write(str(result), f'analytics/{self.name}.csv')

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class DeepQPlayer:
    EPSILON = 0

    def __init__(self, left=False, right=False, brain=None):
        # import necessary modules from keras
        self.left = left
        self.right = right
        self.new_memory = []
        self.game_memory = []
        if brain is None:
            self.brain = DQN()
        else:
            self.brain = brain

    def set_model(self, model):
        self.brain = model

    def move(self, state, debug=False):
        predictions = self.brain.infer(state)
        best = np.argmax(predictions)
        prob = predictions[best]
        if debug:
            print(predictions)
        if random() < DeepQPlayer.EPSILON:
            return choice(Pong.ACTIONS), np.array([1/3, 1/3, 1/3])
        else:
            return Pong.ACTIONS[best], predictions
