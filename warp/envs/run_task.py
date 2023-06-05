import hydra
import yaml

from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from warp.envs import ObjectTask, HandObjectTask, ReposeTask
from warp.envs.utils.common import run_env

OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg in ["", None] else arg)


def custom_eval(x):
    import numpy as np
    import torch

    return eval(x)


OmegaConf.register_new_resolver("eval", custom_eval)


@hydra.main(config_path="cfg", config_name="config.yaml")
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
        env = instantiate(cfg.task.env)
    elif cfg.task.name.lower() == "object_task":
        env = instantiate(cfg.task.env)

    # get a policy
    policy = None
    run_env(env, policy, cfg_full["num_steps"])


if __name__ == "__main__":
    run()
