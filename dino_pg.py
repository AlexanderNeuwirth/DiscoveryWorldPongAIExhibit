from player import PGAgent
from dino import DinoEnv

BATCH_SIZE = 1

if __name__ == "__main__":
    env = DinoEnv()
    state_size = DinoEnv.SCREEN_SIZE[0] * DinoEnv.SCREEN_SIZE[1] // 4
    print(state_size)
    action_size = 3
    agent = PGAgent(state_size=state_size, action_size=action_size, learning_rate=0.01, structure=(300,200,100))

    while True:
        states, actions, probs, rewards = [], [], [], []
        state = DinoEnv.preprocess_screen(env.reset()).flatten()
        for i in range(BATCH_SIZE):
            done = False
            while not done:
                states.append(state)
                choice, prob = agent.act(state)
                print(prob)
                actions.append(choice)
                probs.append(prob)
                action = DinoEnv.ACTIONS[choice]

                state, reward, done = env.step(action)
                rewards.append(reward)
                state = DinoEnv.preprocess_screen(state).flatten()

        agent.train(states, actions, probs, rewards)
        states, actions, probs, rewards = [], [], [], []
