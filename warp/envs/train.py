import traceback
import hydra, os, wandb, yaml, torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from shac.algorithms.shac import SHAC
from shac.algorithms.shac2 import SHAC as SHAC2
from shac.utils.common import *
from warp.envs.utils.rlgames_utils import RLGPUEnvAlgoObserver, RLGPUEnv
from warp import envs
from gym import wrappers
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from warp.envs.utils import hydra_resolvers


def register_envs(env_config, env_type="warp"):
    def create_dflex_env(**kwargs):
        # create env without grads since PPO doesn't need them
        env = instantiate(env_config.config, no_grad=True)

        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)
        return env

    def create_warp_env(**kwargs):
        # create env without grads since PPO doesn't need them
        env = instantiate(env_config.env, no_grad=True)

        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)

        return env

    if env_type == "dflex":
        vecenv.register(
            "DFLEX",
            lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "dflex",
            {
                "env_creator": lambda **kwargs: create_dflex_env(**kwargs),
                "vecenv_type": "DFLEX",
            },
        )
    if env_type == "warp":
        vecenv.register(
            "WARP",
            lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "warp",
            {
                "env_creator": lambda **kwargs: create_warp_env(**kwargs),
                "vecenv_type": "WARP",
            },
        )


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    env_name = job_config["env"]["config"]["_target_"].split(".")[-1]
    try:
        alg_name = job_config["alg"]["_target_"].split(".")[-1]
    except:
        alg_name = job_config["alg"]["name"].upper()
    try:
        # Multirun config
        job_id = HydraConfig().get().job.num
        name = f"{alg_name}_{env_name}_sweep_{job_id}"
        notes = wandb_cfg.get("notes", None)
    except:
        # Normal (singular) run config
        name = f"{alg_name}_{env_name}"
        notes = wandb_cfg["notes"]  # force user to make notes
    return wandb.init(
        project=wandb_cfg.project,
        config=job_config,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )


cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, "cfg")


@hydra.main(config_path="cfg", config_name="train.yaml")
def train(cfg: DictConfig):
    if cfg.debug:
        import warp as wp

        wp.config.mode = "debug"
        wp.config.verify_cuda = True
        wp.config.print_launches = True

    torch.set_default_dtype(torch.float32)
    try:
        cfg_full = OmegaConf.to_container(cfg, resolve=True)
        cfg_yaml = yaml.dump(cfg_full["alg"])
        resume_model = cfg.resume_model
        if os.path.exists("exp_config.yaml"):
            loaded_config = yaml.load(open("exp_config.yaml", "r"))
            params, wandb_id = loaded_config["params"], loaded_config["wandb_id"]
            if cfg.general.run_wandb:
                run = create_wandb_run(cfg.wandb, params, wandb_id)
            resume_model = "restore_checkpoint.zip"
            assert os.path.exists(resume_model), "restore_checkpoint.zip does not exist!"
        else:
            defaults = HydraConfig.get().runtime.choices

            params = yaml.safe_load(cfg_yaml)
            params["defaults"] = {k: defaults[k] for k in ["alg"]}

            if cfg.general.run_wandb:
                run = create_wandb_run(cfg.wandb, params)
            else:
                run = None
            # wandb_id = run.id if run != None else None
            save_dict = dict(wandb_id=run.id if run != None else None, params=params)
            yaml.dump(save_dict, open("exp_config.yaml", "w"))
            print("Alg Config:")
            print(cfg_yaml)
            print("Task Config:")
            print(yaml.dump(cfg_full["task"]))

        if "_target_" in cfg.alg:
            # Run with hydra
            cfg.task.env.no_grad = not cfg.general.train

            traj_optimizer = instantiate(cfg.alg, env_config=cfg.task.env, logdir=cfg.general.logdir)

            if cfg.general.train:
                traj_optimizer.train()
            else:
                traj_optimizer.play(cfg_full)

        elif "shac" in cfg.alg.name:
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
            if not cfg.general.play:
                traj_optimizer.train()
            else:
                traj_optimizer.play(cfg_train)
            wandb.finish()
        elif cfg.alg.name in ["ppo", "sac"]:
            cfg_train = cfg_full["alg"]
            cfg_train["params"]["general"] = cfg_full["general"]
            cfg_train["params"]["seed"] = cfg_full["general"]["seed"]
            cfg_train["params"]["render"] = cfg_full["render"]

            env_name = cfg_train["params"]["config"]["env_name"]
            cfg_train["params"]["diff_env"] = cfg_full["task"]["env"]
            assert cfg_train["params"]["diff_env"].get("no_grad", True), "diffsim should be disabled for ppo"
            cfg_train["params"]["diff_env"]["use_graph_capture"] = True
            cfg_train["params"]["diff_env"]["use_autograd"] = True
            # env_name = cfg_train["params"]["diff_env"].pop("_target_").split(".")[-1]
            cfg_train["params"]["config"]["env_name"] = "warp"

            # save config
            if cfg_train["params"]["general"]["train"]:
                log_dir = cfg_train["params"]["general"]["logdir"]
                os.makedirs(log_dir, exist_ok=True)
                # save config
                yaml.dump(cfg_train, open(os.path.join(log_dir, "cfg.yaml"), "w"))
            # register envs
            register_envs(cfg.task, env_type="warp")

            # add observer to score keys
            if cfg_train["params"]["config"].get("score_keys"):
                algo_observer = RLGPUEnvAlgoObserver()
            else:
                algo_observer = None
            runner = Runner(algo_observer)
            runner.load(cfg_train)
            runner.reset()
            runner.run(cfg_train["params"]["general"])
    except:
        traceback.print_exc(file=open("exception.log", "w"))
        with open("exception.log", "r") as f:
            print(f.read())


if __name__ == "__main__":
    train()
