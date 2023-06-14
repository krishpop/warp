import os
import numpy as np
from warp.envs.environment import RenderMode
from imageio import get_writer


class Monitor:
    def __init__(self, env, save_dir, ep_filter=None):
        self.env = env
        __import__("ipdb").set_trace()
        self.writer = None
        self.save_dir = save_dir or "./videos/"
        print("saving videos to", self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.ep_filter = ep_filter
        self.num_episodes = 0

    def reset(self, *args, **kwargs):
        ret = self.env.reset(*args, **kwargs)
        # self.env.renderer.move_camera(np.zeros(3), 5, 225, -20)  # resets default camera pose
        if self.writer:
            self.writer.close()
        if self.ep_filter is None or self.ep_filter(self.num_episodes):
            self.writer = get_writer(
                os.path.join(self.save_dir, f"ep-{self.num_episodes}.mp4"), fps=int(1 / self.env.frame_dt)
            )
        else:
            self.writer = None
        self.num_episodes += 1
        return ret

    def step(self, action):
        res = self.env.step(action)
        if self.writer is not None:
            self.render()
        return res

    def render(self):
        if self.writer is None:
            return
        img = self.env.render(mode="rgb_array")
        self.writer.append_data((255 * img).astype(np.uint8))
        return

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def close(self):
        self.env.close()
        if self.writer is not None:
            self.writer.close()
