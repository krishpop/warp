import torch
import imageio
import sys
from tqdm import trange
sys.path.append("../")
from envs import HumanoidEnv, HopperEnv, RenderMode

def render_humanoid():
    env = HumanoidEnv(render=True)
    env.reset()
    with imageio.get_writer("test_render.mp4", fps=30) as writer:
        for i in trange(100):
            ac = torch.stack([torch.tensor(env.action_space.sample()) for _ in range(env.num_envs)]).to(torch.float32).cuda()
            env.step(ac)
            writer.append_data(env.render())

    env.renderer.app.window.set_request_exit()


def render_hopper():
    env = HopperEnv(render=True, num_envs=2)
    env.reset()
    tiny_renderer = env.render_mode == RenderMode.TINY
    writer = None
    # writer = imageio.get_writer("test_render.mp4", fps=30) if tiny_renderer else None
    for i in trange(100):
        ac = torch.stack([torch.tensor(env.action_space.sample()) for _ in range(env.num_envs)]).to(torch.float32).cuda()
        env.step(ac)
        img = env.render()
        if writer is not None: writer.append_data(img)
    if writer is not None: writer.close()
    if tiny_renderer:
        env.renderer.app.window.set_request_exit()


def render_humanoid_live():
    HumanoidEnv.tiny_render_settings['mode'] = 'live'  # render as images
    env = HumanoidEnv(render=True)
    try:
        env.run()
    except Exception as e:
        print("exiting run() with exception: ", e)
        env.renderer.app.window.set_request_exit()
        return

    env.renderer.app.window.set_request_exit()

if __name__ == "__main__":
    render_hopper()
    sys.exit(0)
