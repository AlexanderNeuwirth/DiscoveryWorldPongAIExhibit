import numpy as np
from pong import Pong
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
import os
#from player import HumanPlayer
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
        #model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        model.add(Dense(500, activation='relu', init='he_uniform', input_shape=(self.state_size,)))
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
    os.makedirs("models/1", exist_ok=True)
    os.makedirs("models/2", exist_ok=True)
    start_index = None
    actions = ["UP", "DOWN"]
    #env = gym.make("Pong-v0")
    env = Pong()
    state = env.reset()
    prev_x = None
    score_1 = 0
    score_2 = 0
    state_size = Pong.HEIGHT//2 * Pong.WIDTH//2
    action_size = 2 #env.action_space.n
    agent1 = PGAgent(state_size, action_size)
    agent2 = PGAgent(state_size, action_size)
    episode = 0
    if start_index is not None:
        episode = start_index
        agent1.load(f'./models/1/{start_index}.h5')
        agent2.load(f'./models/2/{start_index}.h5')
    #player = HumanPlayer('w', 's')
    last_action_1 = None
    last_action_2 = None
    i = 0
    while True:
        #env.show(4)

        x = preprocess_pong(state)
        #x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        if last_action_1 is None or last_action_2 is None or i % 3 == 0:
            action1, prob1 = agent1.act(x)
            action2, prob2 = agent2.act(x)
            last_action_1 = action1
            last_action_2 = action2
            state, reward, done = env.step(actions[action2], actions[action1])
            reward_1 = float(reward[1])
            reward_2 = float(reward[0])
            agent1.memorize(x, action1, prob1, reward_1)
            agent2.memorize(x, action2, prob2, reward_2)
        else:
            state, reward, done = env.step(actions[last_action_2], actions[last_action_1])
            reward_1 = float(reward[1])
            reward_2 = float(reward[0])
        if reward_1 > 0: score_1 += reward_1
        elif reward_2 > 0: score_2 += reward_2

        i += 1
        if done:
            i = 0

            episode += 1
            agent1.train()
            agent2.train()
            print('Episode: %d - Score: %f - %f.' % (episode, score_1, score_2))
            score_1 = 0
            score_2 = 0
            state = env.reset()
            prev_x = None
            if episode > 1 and episode % 50 == 0:
                agent1.save(f'./models/1/{episode}.h5')
                agent2.save(f'./models/2/{episode}.h5')
