from player import PGAgent
from dino import DinoEnv
import time

BATCH_SIZE = 1

if __name__ == "__main__":
    env = DinoEnv()
    state_size = (DinoEnv.SCREEN_SIZE[0] // 4, DinoEnv.SCREEN_SIZE[1] // 4, 1)
    print(state_size)
    action_size = 3
    agent = PGAgent(state_size=state_size, action_size=action_size, learning_rate=0.0001, structure='conv')
    generation = 0
    while True:
        states, actions, probs, rewards = [], [], [], []
        state = DinoEnv.preprocess_screen(env.reset())
        for i in range(BATCH_SIZE):
            done = False
            while not done:
                states.append(state)
                print(f'state: {state.shape}')
                choice, prob = agent.act(state)
                actions.append(choice)
                probs.append(prob)
                action = DinoEnv.ACTIONS[choice]

                state, reward, done = env.step(action)
                rewards.append(reward)
                state = DinoEnv.preprocess_screen(state)
            print("Score: " + str(env.get_score()))
        agent.train(states, actions, probs, rewards)
        if(generation % 100 == 0):
            agent.save(str(generation) + "_" + str(time.time()) + ".h5");
        states, actions, probs, rewards = [], [], [], []
        generation += 1
