# coding: utf-8
from shac.envs.hopper import HopperEnv
import torch
from tqdm import trange

env = HopperEnv(num_envs=2, render=True, episode_length=1000, stochastic_init=False)
env.reset()
__import__("ipdb").set_trace()

print("before", env.contact_count_changed)

done = False

# while not done:
for _ in trange(100):
    ac = torch.randn(
        (env.num_envs, env.num_acts), device=env.device, dtype=torch.float32
    )
    obs, rew, done, info = env.step(ac)
    env.render()
    done = done.all()

print("after", env.contact_count_changed)
