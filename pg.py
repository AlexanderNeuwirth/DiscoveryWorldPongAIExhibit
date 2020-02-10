import os
from utils import save_video, plot_loss, plot_score, write
import simulator
from player import PGAgent, BotPlayer
from visualizer import get_weight_image

GAME_BATCH = 10

if __name__ == "__main__":
    os.makedirs("models/l", exist_ok=True)
    os.makedirs("models/r", exist_ok=True)
    os.makedirs("analytics", exist_ok=True)
    os.makedirs("analytics/plots", exist_ok=True)
    start_index = None

    agent_l = BotPlayer(left=True)
    agent_r = PGAgent(simulator.CUSTOM_STATE_SIZE, simulator.CUSTOM_ACTION_SIZE, "agent_r")

    episode = 0
    if start_index is not None:
        episode = start_index
        #agent_l.load(f'./models/l/{start_index}.h5')
        agent_r.load(f'./models/r/{start_index}.h5')
    neuron_states = []
    while True:
        episode += 1
        states, left, right, meta = simulator.simulate_game(simulator.CUSTOM, left=agent_l, right=agent_r, batch=1)
        render_states, model_states, (score_l, score_r) = meta
        actions, probs, rewards = right
        #agent_l.train(states, *left)
        agent_r.train(states, *right)
        neuron_states.append(get_weight_image(agent_r.model))
        write(f'{score_l},{score_r}', f'analytics/scores.csv')
        if episode == 1 or episode % 5 == 0:
            save_video(render_states, f'./analytics/{episode}.mp4')
            save_video(neuron_states, f'./analytics/{episode}_weights0.mp4')
            plot_loss(f'./analytics/plots/loss_{episode}.png')
            plot_score(f'./analytics/plots/score_{episode}.png')
            #agent_l.save(f'./models/l/{episode}.h5')
            agent_r.save(f'./models/r/{episode}.h5')
        if episode == 10000:
            exit(0)
