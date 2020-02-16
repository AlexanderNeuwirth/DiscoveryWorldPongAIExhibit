import os
import simulator
from utils import save_video, plot_loss, plot_score, write
from player import PGAgent, BotPlayer
from visualizer import get_weight_image

GAME_BATCH = 10
MODE = simulator.HIT_PRACTICE

if __name__ == "__main__":
    # Ensure directory safety
    os.makedirs("models/l", exist_ok=True)
    os.makedirs("models/r", exist_ok=True)
    os.makedirs("analytics", exist_ok=True)
    os.makedirs("analytics/plots", exist_ok=True)

    # Initialize for checks & scope
    start_index = None
    agent_l = None
    state_size = None
    state_shape = None
    action_size = None

    # Set constants based on env setting
    if MODE == simulator.CUSTOM or MODE == simulator.HIT_PRACTICE:
        action_size = simulator.CUSTOM_ACTION_SIZE
        state_size = simulator.CUSTOM_STATE_SIZE
        state_shape = simulator.CUSTOM_STATE_SHAPE
        agent_l = BotPlayer(left=True) if MODE == simulator.CUSTOM else None # Default to bot, override with model if needed
    if MODE == simulator.ATARI:
        action_size = simulator.ATARI_ACTION_SIZE
        state_size = simulator.ATARI_STATE_SIZE
        state_shape = simulator.ATARI_STATE_SHAPE

    # Init agent
    agent_r = PGAgent(state_size, action_size, "agent_r")

    # Type checks for convenience later
    r_is_model = type(agent_r) == PGAgent
    l_is_model = type(agent_l) == PGAgent

    episode = 0
    # Optional checkpoint loading
    if start_index is not None:
        episode = start_index
        if l_is_model: agent_l.load(f'./models/l/{start_index}.h5')
        agent_r.load(f'./models/r/{start_index}.h5')

    # Store neuron images for fun
    neuron_states = []

    # Train loop
    while True:
        episode += 1
        states, left, right, meta = simulator.simulate_game(MODE, left=agent_l, right=agent_r, batch=GAME_BATCH)
        render_states, model_states, (score_l, score_r) = meta
        actions, probs, rewards = right

        if r_is_model: agent_r.train(states, *right)
        if l_is_model: agent_l.train(states, *left)

        neuron_states.append(get_weight_image(agent_r.model, size=state_shape))
        if episode == 1 or episode % 50 == 0:
            save_video(render_states, f'./analytics/{episode}.mp4')
            if r_is_model: save_video(neuron_states, f'./analytics/{episode}_weights0.mp4', fps=60)
            plot_loss(f'./analytics/plots/loss_{episode}.png', include_left=False)
            plot_score(f'./analytics/plots/score_{episode}.png')
            if l_is_model: agent_l.save(f'./models/l/{episode}.h5')
            if r_is_model: agent_r.save(f'./models/r/{episode}.h5')
        if episode == 10000:
            exit(0)
