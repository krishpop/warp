import hydra
import torch
import yaml

from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from warp.envs import ObjectTask, HandObjectTask, ReposeTask
from warp.envs.utils.common import run_env, HandType

# register custom resolvers
import warp.envs.utils.hydra_resolvers


def get_policy(cfg):
    if cfg.alg is None or cfg.alg.name == "default":
        return None
    if cfg.alg.name == "random":
        num_act = 16 if cfg.task.env.hand_type == HandType.ALLEGRO else 24
        return lambda x, t: torch.rand((x.shape[0], num_act), device=x.device).clamp_(-1.0, 1.0)

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

    # get a policy
    policy = get_policy(cfg)
    run_env(env, policy, cfg_full["num_steps"])


if __name__ == "__main__":
    run()
