import hydra
import yaml

from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from warp.envs import ObjectTask, HandObjectTask, ReposeTask
from warp.envs.utils.common import run_env

# register custom resolvers
import warp.envs.utils.hydra_resolvers


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
        env = instantiate(cfg.task.env)
    elif cfg.task.name.lower() == "object_task":
        env = instantiate(cfg.task.env)

    # get a policy
    policy = None
    run_env(env, policy, cfg_full["num_steps"])


if __name__ == "__main__":
    run()
