import hydra
import torch
import yaml
import os
import numpy as np

from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from shac.utils.rlgames_utils import RLGPUEnvAlgoObserver, RLGPUEnv
from shac.algorithms.shac import SHAC
from shac.algorithms.shac2 import SHAC as SHAC2
from rl_games.torch_runner import Runner
from warp.envs import ObjectTask, HandObjectTask, ReposeTask
from warp.envs.utils.common import run_env, HandType, ActionType, get_time_stamp
from warp.envs.train import register_envs
from warp.envs.wrappers import Monitor

# register custom resolvers
import warp.envs.utils.hydra_resolvers


def get_policy(cfg):
    if cfg.alg is None or cfg.alg.name == "default":
        return None
    num_act = 16 if cfg.task.env.hand_type == HandType.ALLEGRO else 24
    if cfg.alg.name == "zero":
        return lambda x, t: torch.zeros((x.shape[0], num_act), device=x.device)
    if cfg.alg.name == "random":
        return lambda x, t: torch.rand((x.shape[0], num_act), device=x.device).clamp_(-1.0, 1.0)
    if cfg.alg.name == "sine":
        return lambda x, t: torch.sin(torch.ones((x.shape[0], num_act)).to(x) * t * 2 * np.pi * 0.1)
    


@hydra.main(config_path="cfg", config_name="run_task.yaml")
def run(cfg: DictConfig):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)
    cfg_yaml = yaml.dump(cfg_full)
    # params = yaml.safe_load(cfg_yaml)
    print("Run Params:")
    print(cfg_yaml)

    # instantiate the environment
    if cfg.task.name.lower() == "repose_task":
        env = instantiate(cfg.task.env, _convert_="partial")
    elif cfg.task.name.lower() == "hand_object_task":
        env = instantiate(cfg.task.env, _convert_="partial")
    elif cfg.task.name.lower() == "object_task":
        env = instantiate(cfg.task.env, _convert_="partial")

    if env.opengl_render_settings.get('headless', False):
        env = Monitor(env, "outputs/videos/{}".format(get_time_stamp()))

    # get a policy
    if cfg.alg.name in ["default", "random", "zero", "sine"]:
        policy = get_policy(cfg)
        run_env(env, policy, cfg_full["num_steps"], cfg_full["num_rollouts"])
    elif cfg.alg.name in ["ppo", "sac"]:
        cfg_eval = cfg_full["alg"]
        cfg_eval["params"]["general"] = cfg_full["general"]
        cfg_eval["params"]["seed"] = cfg_full["general"]["seed"]
        cfg_eval["params"]["render"] = cfg_full["render"]
        cfg_eval["params"]["diff_env"] = cfg_full["task"]["env"]
        env_name = cfg_full["task"]["name"]
        register_envs(cfg_eval, env_name)
        # add observer to score keys
        if cfg_eval["params"]["config"].get("score_keys"):
            algo_observer = RLGPUEnvAlgoObserver()
        else:
            algo_observer = None
        runner = Runner(algo_observer)
        runner.load(cfg_eval)
        runner.reset()
        runner.run(cfg_eval["params"]["general"])
    elif cfg.alg.name in ["shac", "shac2"]:
        if cfg.alg.name == "shac2":
            traj_optimizer = SHAC2(cfg)
        elif cfg.alg.name == "shac":
            cfg_train = cfg_full["alg"]
            if cfg.general.play:
                cfg_train["params"]["config"]["num_actors"] = (
                    cfg_train["params"]["config"].get("player", {}).get("num_actors", 1)
                )
            if not cfg.general.no_time_stamp:
                cfg.general.logdir = os.path.join(cfg.general.logdir, get_time_stamp())

            cfg_train["params"]["general"] = cfg_full["general"]
            cfg_train["params"]["render"] = cfg_full["render"]
            cfg_train["params"]["general"]["render"] = cfg_full["render"]
            cfg_train["params"]["diff_env"] = cfg_full["task"]["env"]
            env_name = cfg_train["params"]["diff_env"].pop("_target_")
            cfg_train["params"]["diff_env"]["name"] = env_name.split(".")[-1]
            # TODO: Comment to disable autograd/graph capture for diffsim
            # cfg_train["params"]["diff_env"]["use_graph_capture"] = False
            # cfg_train["params"]["diff_env"]["use_autograd"] = True
            print(cfg_train["params"]["general"])
            traj_optimizer = SHAC(cfg_train)
        traj_optimizer.play(cfg_train)


if __name__ == "__main__":
    run()
