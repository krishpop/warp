import traceback
import hydra, os, wandb, yaml, torch
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from shac.algorithms.shac import SHAC
from shac.algorithms.shac2 import SHAC as SHAC2
from shac.utils.common import *
from warp.envs.utils.rlgames_utils import RLGPUEnvAlgoObserver, RLGPUEnv
import warp as wp
from warp import envs
from gym import wrappers
from hydra.utils import instantiate
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from warp.envs.utils import hydra_resolvers
from svg.train import Workspace
from omegaconf import OmegaConf, open_dict


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
        env = instantiate(env_config.config, no_grad=True)

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


# def create_wandb_run(wandb_cfg, job_config, run_id=None):
#     """
#     Creates and initializes a run in Weights & Biases (wandb).

#     Args:
#         wandb_cfg (dict): Configuration for wandb.
#         job_config (dict): Configuration for the job.
#         run_id (str, optional): ID of the run to resume. Defaults to None.

#     Returns:
#         WandbRun: The initialized wandb run.
#     """
#     # Get environment name from job_config
#     env_name = job_config["task"]["env"]["_target_"].split(".")[-1]

#     try:
#         # Get algorithm name from job_config
#         alg_name = job_config["alg"]["_target_"].split(".")[-1]
#     except:
#         # Use default algorithm name if not found in job_config
#         alg_name = "PPO"

#     try:
#         # Multirun config
#         job_id = HydraConfig().get().job.num
#         override_dirname = HydraConfig().get().job.override_dirname
#         name = f"{wandb_cfg.sweep_name_prefix}-{job_id}"
#         notes = f"{override_dirname}"
#     except:
#         # Normal (singular) run config
#         name = f"{alg_name}_{env_name}"
#         notes = wandb_cfg["notes"]  # force user to make notes

#     return wandb.init(
#         project=wandb_cfg.project,
#         config=job_config,
#         group=wandb_cfg.group,
#         entity=wandb_cfg.entity,
#         sync_tensorboard=True,
#         monitor_gym=True,
#         save_code=True,
#         name=name,
#         notes=notes,
#         id=run_id,
#         resume=run_id is not None,
#     )


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
    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    if cfg.general.run_wandb:
        create_wandb_run(cfg.wandb, cfg_full)

    # patch code to make jobs log in the correct directory when doing multirun
    logdir = HydraConfig.get()["runtime"]["output_dir"]
    logdir = os.path.join(logdir, cfg.general.logdir)

    torch.set_default_dtype(torch.float32)

    if cfg.debug:
        wp.config.mode = "debug"
        wp.config.verify_cuda = True
        wp.config.print_launches = True

    # try:
    #     cfg_yaml = yaml.dump(cfg_full["alg"])
    #     resume_model = cfg.resume_model
    #     run = None
    #     if os.path.exists("exp_config.yaml"):
    #         loaded_config = yaml.load(open("exp_config.yaml", "r"))
    #         params, wandb_id = loaded_config["params"], loaded_config["wandb_id"]
    #         if cfg.general.run_wandb:
    #             run = create_wandb_run(cfg.wandb, params, wandb_id)
    #             resume_model = "restore_checkpoint.zip"
    #             assert os.path.exists(resume_model), "restore_checkpoint.zip does not exist!"
    #     else:
    #         defaults = HydraConfig.get().runtime.choices

    #         params = yaml.safe_load(cfg_yaml)
    #         params["defaults"] = {k: defaults[k] for k in ["alg"]}
    #         if cfg.general.run_wandb:
    #             run = create_wandb_run(cfg.wandb, params)
    #         # wandb_id = run.id if run != None else None
    #         save_dict = dict(wandb_id=run.id if run != None else None, params=params)
    #         yaml.dump(save_dict, open("exp_config.yaml", "w"))
    #         print("Alg Config:")
    #         print(cfg_yaml)
    #         print("Task Config:")
    #         print(yaml.dump(cfg_full["task"]))

    #     logdir = HydraConfig.get()["runtime"]["output_dir"]
    #     logdir = os.path.join(logdir, cfg.general.logdir)

    if "_target_" in cfg.alg:
        # Run with hydra
        cfg.env.config.no_grad = not cfg.general.train

        # if "AHAC" in cfg.alg._target_ and "jacobian_norm" in cfg.env.ahac:
        #     with open_dict(cfg):
        #         cfg.env.config.jacobian_norm = cfg.env.ahac.jacobian_norm

        score_keys = []
        try:
            score_keys = cfg.env.score_keys
        except:
            pass

        traj_optimizer = instantiate(cfg.alg, env_config=cfg.env.config, logdir=logdir, score_keys=score_keys)

        if cfg.general.checkpoint:
            traj_optimizer.load(cfg.general.checkpoint)

        if cfg.general.train:
            traj_optimizer.train()
        else:
            traj_optimizer.run(cfg.env.player.games_num)

    # elif "shac" in cfg.alg.name:
    #     if cfg.alg.name == "shac2":
    #         traj_optimizer = SHAC2(cfg)
    #     elif cfg.alg.name == "shac":
    #         cfg_train = cfg_full["alg"]
    #         if cfg.general.play:
    #             cfg_train["params"]["config"]["num_actors"] = (
    #                 cfg_train["params"]["config"].get("player", {}).get("num_actors", 1)
    #             )
    #         if not cfg.general.no_time_stamp:
    #             cfg.general.logdir = os.path.join(cfg.general.logdir, get_time_stamp())

    #         cfg_train["params"]["general"] = cfg_full["general"]
    #         cfg_train["params"]["render"] = cfg_full["render"]
    #         cfg_train["params"]["general"]["render"] = cfg_full["render"]
    #         cfg_train["params"]["diff_env"] = cfg_full["task"]["env"]
    #         env_name = cfg_train["params"]["diff_env"].pop("_target_")
    #         cfg_train["params"]["diff_env"]["name"] = env_name.split(".")[-1]
    #         # TODO: Comment to disable autograd/graph capture for diffsim
    #         # cfg_train["params"]["diff_env"]["use_graph_capture"] = False
    #         # cfg_train["params"]["diff_env"]["use_autograd"] = True
    #         print(cfg_train["params"]["general"])
    #         traj_optimizer = SHAC(cfg_train)
    #     if not cfg.general.play:
    #         traj_optimizer.train()
    #     else:
    #         traj_optimizer.play(cfg_train)
    #     wandb.finish()
    elif cfg.alg.name in ["ppo", "sac"]:
        # if not hydra init, then we must have PPO
        # to set up RL games we have to do a bunch of config menipulation
        # which makes it a huge mess...

        # PPO doesn't need env grads
        cfg.env.config.no_grad = True

        # first shuffle around config structure
        cfg_train = cfg_full["alg"]
        cfg_train["params"]["general"] = cfg_full["general"]
        env_name = cfg_train["params"]["config"]["env_name"]
        cfg_train["params"]["diff_env"] = cfg_full["env"]["config"]
        cfg_train["params"]["general"]["logdir"] = logdir

        # boilerplate to get rl_games working
        cfg_train["params"]["general"]["play"] = not cfg_train["params"]["general"]["train"]

        # Now handle different env instantiation
        if env_name.split("_")[0] == "df":
            cfg_train["params"]["config"]["env_name"] = "dflex"
        elif env_name.split("_")[0] == "warp":
            cfg_train["params"]["config"]["env_name"] = "warp"
        env_name = cfg_train["params"]["diff_env"]["_target_"]
        cfg_train["params"]["diff_env"]["name"] = env_name.split(".")[-1]

        # cfg_train["params"]["diff_env"] = cfg_full["task"]["env"]
        # assert cfg_train["params"]["diff_env"].get("no_grad", True), "diffsim should be disabled for ppo"
        # # cfg_train["params"]["diff_env"]["use_graph_capture"] = True
        # # cfg_train["params"]["diff_env"]["use_autograd"] = True
        # # env_name = cfg_train["params"]["diff_env"].pop("_target_").split(".")[-1]
        # cfg_train["params"]["diff_env"]["name"] = env_name

        # save config
        if cfg_train["params"]["general"]["train"]:
            os.makedirs(logdir, exist_ok=True)
            yaml.dump(cfg_train, open(os.path.join(logdir, "cfg.yaml"), "w"))

        # register envs with the correct number of actors for PPO
        if cfg.alg.name == "ppo":
            cfg["env"]["config"]["num_envs"] = cfg["env"]["ppo"]["num_actors"]
        else:
            cfg["env"]["config"]["num_envs"] = cfg["env"]["sac"]["num_actors"]
        register_envs(cfg.env)

        # add observer to score keys
        if cfg_train["params"]["config"].get("score_keys"):
            algo_observer = RLGPUEnvAlgoObserver()
        else:
            algo_observer = None
        runner = Runner(algo_observer)
        runner.load(cfg_train)
        runner.reset()
        if cfg.render:
            cfg_train["params"]["render"] = True
        player = runner.run(cfg_train["params"]["general"])
        # if running first with train=True with render=True
        if cfg.general.train:
            cfg_train["params"]["render"] = True
            runner.load_config(cfg_train)
            player = runner.run_play(cfg_train["params"]["general"])
            player.env.renderer.save()
        else:
            player.env.renderer.save()

    elif cfg.alg.name == "svg":
        cfg.env.config.no_grad = True
        with open_dict(cfg):
            cfg.alg.env = cfg.env.config
        w = Workspace(cfg.alg)
        w.run_epochs()
    else:
        raise NotImplementedError

    if cfg.general.run_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
