import time
import numpy as np
import os

class Timer:
    timers = {}

    @staticmethod
    def start(name):
        Timer.timers[name] = time.time()

    @staticmethod
    def stop(name):
        t = time.time() - Timer.timers[name]
        print(f'Finished {name} in {round(t, 5)}s')


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def encode_action(action):
    if action == "UP":
        return np.asarray([1, 0], dtype=np.float32)
    elif action == "DOWN":
        return np.asarray([0, 1], dtype=np.float32)
    elif action == "NONE":
        return np.asarray([0, 0], dtype=np.float32)
    else:
        raise NameError(f"Action {action} does not exist")


def encode_gradient(action, prob):
    if action == "UP":
        return np.asarray([1, 0], dtype=np.float32) - prob
    elif action == "DOWN":
        return np.asarray([0, 1], dtype=np.float32) - prob
    elif action == "NONE":
        return np.asarray([0, 0], dtype=np.float32) - prob
    else:
        raise NameError(f"Action {action} does not exist")


def normalize_states(states):
    states /= 255
    return states


def get_last_file():
    id = get_resume_index()
    if id:
        return os.path.join("models", f"{id}.h5")
    else:
        return None

"""
def save_video(states, path):
    clips = [moviepy.editor.ImageClip(m).set_duration(0.1)
             for m in states]

    concat_clip = moviepy.editor.concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(path, fps=60)
"""

def write(value, file):
    text_file = open(file, "a+")
    text_file.write(value + '\n')
    text_file.close()


def get_resume_index():
    files = [f for f in os.listdir("models") if os.path.isfile(os.path.join("models", f))]
    ids = [int(os.path.split(f)[1].split('.')[0]) for f in files]
    max = -1
    for id in ids:
        if id > max:
            max = id
    if max == -1:
        return None
    return max
