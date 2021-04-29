from exhibit.ai.model import PGAgent
from exhibit.shared.config import Config
import time
from exhibit.ai.ai_subscriber import AISubscriber
import numpy as np


class AIDriver:
    def publish_inference(self):
        # Get latest state diff
        diff_state = self.state.render_latest_diff()
        current_frame_id = self.state.frame
        import cv2
        cv2.imshow("test", diff_state)
        cv2.waitKey(1)
        # Infer on flattened state vector
        x = diff_state.ravel()
        action, probs = self.agent.act(x)

        # Publish prediction
        self.state.publish("paddle2/action", {"action": str(action)})
        self.state.publish("paddle2/frame", {"frame": current_frame_id})
        print(current_frame_id)

        if len(self.frame_diffs) > 1000:
            print(
                f"Frame distribution: mean {np.mean(self.frame_diffs)}, stdev {np.std(self.frame_diffs)} counts {np.unique(self.frame_diffs, return_counts=True)}")
            self.frame_diffs = []

    def __init__(self):
        self.agent = PGAgent(Config.CUSTOM_STATE_SIZE, Config.CUSTOM_ACTION_SIZE)
        self.agent.load('../../validation/6px_7k.h5')
        self.state = AISubscriber(trigger_event=lambda: self.publish_inference())
        self.last_frame_id = self.state.frame
        self.last_tick = time.time()
        self.frame_diffs = []
        self.state.start()


if __name__ == "__main__":
    instance = AIDriver()
