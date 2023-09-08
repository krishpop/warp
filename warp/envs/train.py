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


def register_envs(cfg_train, env_name):
    def create_warp_env(**kwargs):
        env_fn = getattr(envs, cfg_train["params"]["diff_env"]["_target_"].split(".")[-1])
        env_kwargs = kwargs
        skip_keys = [
            "_target_",
            "num_envs",
            "render",
            "seed",
            "episode_length",
            "no_grad",
            "stochastic_init",
            "name",
            "env_name",
        ]
        env_kwargs.update({k: v for k, v in cfg_train["params"]["diff_env"].items() if k not in skip_keys})
        if cfg_train["params"].get("seed", None):
            seed = cfg_train["params"]["seed"]
            env_kwargs.pop("seed", None)
        else:
            seed = env_kwargs.pop("seed", 42)

        env = env_fn(
            num_envs=cfg_train["params"]["config"]["num_actors"],
            render=cfg_train["params"]["render"],
            seed=seed,
            episode_length=cfg_train["params"]["diff_env"].get("episode_length", 1000),
            no_grad=True,
            stochastic_init=cfg_train["params"]["diff_env"]["stochastic_init"],
            **env_kwargs,
        )

        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)

        return env

    vecenv.register(
        "WARP",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        env_name,
        {"env_creator": lambda **kwargs: create_warp_env(**kwargs), "vecenv_type": "WARP"},
    )


def create_wandb_run(wandb_cfg, job_config, run_id=None, run_wandb=False):
    try:
        job_id = HydraConfig().get().job.num
        override_dirname = HydraConfig().get().job.override_dirname
        name = f"{wandb_cfg.sweep_name_prefix}-{job_id}"
        notes = f"{override_dirname}"
    except:
        name, notes = None, None
    if run_wandb:
        return wandb.init(
            entity=wandb_cfg.entity,
            project=wandb_cfg.project,
            config=job_config,
            group=wandb_cfg.group,
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
            run = create_wandb_run(cfg.wandb, params, wandb_id, run_wandb=cfg.general.run_wandb)
            resume_model = "restore_checkpoint.zip"
            assert os.path.exists(resume_model), "restore_checkpoint.zip does not exist!"
        else:
            defaults = HydraConfig.get().runtime.choices

            params = yaml.safe_load(cfg_yaml)
            params["defaults"] = {k: defaults[k] for k in ["alg"]}

            run = create_wandb_run(cfg.wandb, params, run_wandb=cfg.general.run_wandb)
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
            cfg_train["params"]["diff_env"]["name"] = env_name

            # save config
            if cfg_train["params"]["general"]["train"]:
                log_dir = cfg_train["params"]["general"]["logdir"]
                os.makedirs(log_dir, exist_ok=True)
                # save config
                yaml.dump(cfg_train, open(os.path.join(log_dir, "cfg.yaml"), "w"))
            # register envs
            register_envs(cfg_train, env_name)

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
