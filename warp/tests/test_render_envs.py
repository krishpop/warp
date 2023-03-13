import torch
import imageio
import sys
from tqdm import trange
from warp.envs import HumanoidEnv

def render_humanoid():
    env = HumanoidEnv(render=True)
    env.reset()
    with imageio.get_writer("test_render.mp4", fps=30) as writer: 
        for i in trange(100):
            ac = torch.stack([torch.tensor(env.action_space.sample()) for _ in range(env.num_envs)]).to(torch.float32).cuda()
            env.step(ac)
            writer.append_data(env.render())

    env.renderer.app.window.set_request_exit()
    sys.exit(0)


if __name__ == "__main__":
    render_humanoid()
