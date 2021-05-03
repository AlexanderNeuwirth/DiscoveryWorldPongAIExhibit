import os
from exhibit.train import simulator
from exhibit.shared.utils import save_video, plot_loss, plot_score
from exhibit.shared.config import Config
from exhibit.game.player import BotPlayer
from exhibit.ai.model import PGAgent
from visualizer import get_weight_image

"""
This file is the driver for training a new DRL pong model.
It brings together the following elements:

* The environment simulator (either the custom one found in pong.py or the Atari emulator provided by OpenAI Gym)
  Both environments are wrapped by the interface in simulator.py
* The two agents (some combination of human-controlled, DRL, and hard-coded agents found in player.py)
The level of abstraction in this file is pretty high, and it really only exists to further abstract the training
process into a few environmental and training hyperparameters that are easy to experiment with and to provide
convenient monitoring and graphing of the training process.
"""

GAME_BATCH = 10
MODE = Config.HIT_PRACTICE
LEARNING_RATE = 0.001
DENSE_STRUCTURE = (200,)
ALWAYS_FOLLOW = False

if __name__ == "__main__":
    # Ensure directory safety
    os.makedirs("models/l", exist_ok=True)
    os.makedirs("models/r", exist_ok=True)
    os.makedirs("analytics", exist_ok=True)
    os.makedirs("analytics/plots", exist_ok=True)

    # Initialize for checks & scope
    start_index = None

    # Set constants for custom env
    action_size = Config.CUSTOM_ACTION_SIZE
    state_size = Config.CUSTOM_STATE_SIZE
    state_shape = Config.CUSTOM_STATE_SHAPE
    agent_l = BotPlayer(left=True,
                        always_follow=ALWAYS_FOLLOW) if MODE == Config.CUSTOM else None  # Default to bot, override with model if needed
    # Switch out for interactive session (against human)
    # from player import HumanPlayer
    # agent_l = HumanPlayer(up='w', down='s')

    # Init agent
    agent_r = PGAgent(state_size, action_size, name="agent_r", learning_rate=LEARNING_RATE, structure=DENSE_STRUCTURE)

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
        states, left, right, meta = simulator.simulate_game(env_type=MODE, left=agent_l, right=agent_r, batch=GAME_BATCH)
        render_states, model_states, (score_l, score_r) = meta
        actions, probs, rewards = right

        if r_is_model: agent_r.train(states, *right)
        if l_is_model: agent_l.train(states, *left)

        neuron_states.append(get_weight_image(agent_r.model, size=state_shape))
        if episode == 1 or episode % 50 == 0:
            save_video(render_states, f'./analytics/{episode}.mp4')
            plot_loss(f'./analytics/plots/loss_{episode}.png', include_left=False)
            plot_score(f'./analytics/plots/score_{episode}.png')
            if l_is_model: agent_l.save(f'./models/l/{episode}.h5')
            if r_is_model: agent_r.save(f'./models/r/{episode}.h5')
        if episode == 10000:
            if r_is_model: save_video(neuron_states, f'./analytics/{episode}_weights0.mp4', fps=60)
            exit(0)