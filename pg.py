import gym
import numpy as np
from player import BotPlayer
from pong import Pong
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
import cv2
from keras.layers.convolutional import Convolution2D


class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Reshape((1, Pong.WIDTH//2, Pong.HEIGHT//2), input_shape=(self.state_size,)))
        model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def memorize(self, state, action, prob, reward):
        #print(f'State: {state.dtype} {state.shape} min {np.amin(state)} max {np.amax(state)} {state}')
        #print(f'Action: {action.dtype} {action.shape} min {np.amin(action)} max {np.amax(action)} {action}')
        #print(f'Prob: {prob.dtype} {prob.shape} min {np.amin(prob)} max {np.amax(prob)} {prob}')
        #print(f'Reward: {type(reward)} {reward}')
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
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

    def train(self):
        gradients = np.vstack(self.gradients)
        #print(gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        #for i in range(len(X)):
        #    cv2.imshow("test", X[i].reshape((Pong.HEIGHT//2, Pong.WIDTH//2)))
        #    print(self.probs[i])
        #    print(Y[i])
        #    cv2.waitKey(0)
        #print(Y)
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def preprocess_pong(I):
    I[I == 255] = 1
    I[I == -255] = -1
    return I.astype(np.float).ravel()

def preprocess(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

if __name__ == "__main__":
    actions = ["UP", "DOWN"]
    #env = gym.make("Pong-v0")
    env = Pong()
    bot = BotPlayer(env, left=True)
    state = env.reset()
    prev_x = None
    score = 0
    episode = 0

    state_size = Pong.HEIGHT//2 * Pong.WIDTH//2
    action_size = 2 #env.action_space.n
    agent = PGAgent(state_size, action_size)
    #agent.load('./models/0.h5')
    last_action = None
    i = 0
    while True:
        #env.show()

        x = preprocess_pong(state)
        #x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        bot_action = bot.move(state)
        if last_action is None or i % 3 == 0:
            action, prob = agent.act(x)
            last_action = action
            state, reward, done = env.step(bot_action, actions[action])
            reward = float(reward[1])
            agent.memorize(x, action, prob, reward)
        else:
            state, reward, done = env.step(bot_action, actions[last_action])
            reward = float(reward[1])
        score += reward

        i += 1
        if done:
            i = 0
            episode += 1
            agent.train()
            print('Episode: %d - Score: %f.' % (episode, score))
            score = 0
            state = env.reset()
            prev_x = None
            if episode > 1 and episode % 50 == 0:
                agent.save(f'./models/{episode}.h5')
