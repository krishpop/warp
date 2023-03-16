import os
from gym import spaces
import torch
import numpy as np
import warp as wp
from dmanip.utils.common import set_seed, to_torch, clear_state_grads, to_numpy
from dmanip.utils import warp_utils as wpu
import warp.torch
import warp.sim
import warp.sim.render
from warp.envs.environment import Environment, RenderMode, IntegratorType


class WarpEnv(Environment):
    dt = 1.0 / 60.0
    sim_substeps = 4
    sim_dt = dt
    env_offset = (6.0, 0.0, 6.0)
    tiny_render_settings = dict(scaling=3.0, mode='rgb')
    usd_render_settings = dict(scaling=100.0)
    use_graph_capture = False

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    xpbd_settings = dict(
        iterations=2,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.5,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )
    activate_ground_plane: bool = True

    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=False,
        device="cuda",
        env_name="warp_env",
        render_mode=RenderMode.USD,
        stage_path=None
    ):
        self.seed = seed
        self.requires_grad = not no_grad
        self.device = str(device)
        self.visualize = render
        self.render_mode = render_mode
        self.stochastic_init = stochastic_init

        print("Running with stochastic_init: ", self.stochastic_init)
        self.sim_time = 0.0
        self.num_frames = 0


        self.num_environments = num_envs
        self.env_name = env_name

        if stage_path is None:
            self.stage_path = f'{self.env_name}_{self.num_envs}'
        else:
            self.stage_path = stage_path

        if self.render_mode == RenderMode.TINY and stage_path:
            self.tiny_render_settings['mode'] = "video"
        elif self.render_mode == RenderMode.USD:
            self.usd_render_settings['path'] = f"outputs/{stage_path}.usd"

        # initialize observation and action space
        self.num_observations = num_obs
        self.num_actions = num_act
        self.episode_length = episode_length

        set_seed(self.seed)

        self.obs_space = spaces.Box(
            np.ones(self.num_observations) * -np.Inf,
            np.ones(self.num_observations) * np.Inf,
        )
        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )

        # allocate buffers
        if self.requires_grad:
            self.obs_buf = torch.zeros(
                (self.num_envs, self.num_observations),
                device=self.device,
                dtype=torch.float,
            )
            self.rew_buf = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions),
            device=self.device,
            dtype=torch.float,
            requires_grad=self.requires_grad,
        )

        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        # end of the episode
        self.termination_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )

        self.extras = {}
        self._model = None

    @property
    def model(self):
        if self._model is None:
            raise NotImplementedError("Model not initialized")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

    def init_sim(self):
        self.init()
        self.initialize_renderer()
        self.state_0 = self.model.state(requires_grad=self.requires_grad)
        self.state_1 = self.model.state(requires_grad=self.requires_grad)

    def initialize_renderer(self, stage_path):
        if stage_path is not None:
            self.stage_path = stage_path
        print("Initializing renderer writing to path: outputs/{}".format(self.stage_path))
        if self.render_mode == RenderMode.USD:
            self.renderer = wp.sim.render.SimRenderer(
                self.model, **self.usd_render_settings
            )
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
        elif self.render_mode == RenderMode.TINY:
            self.renderer = wp.sim.tiny_render.TinyRenderer(
                self.model,
                self.env_name,  # window name
                upaxis=self.upaxis,
                env_offset=self.env_offset,
                **self.tiny_render_settings)

        self.render_time = 0.0
        self.render_freq = 10

    def calculateObservations(self):
        """
        Calculate the observations for the current state
        """
        raise NotImplementedError

    def calculateReward(self):
        """
        Calculate the reward for the current state
        """
        raise NotImplementedError

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        """
        Get the rand initial state for the environment
        """
        raise NotImplementedError

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = np.arange(self.num_envs, dtype=int)
        if env_ids is not None:
            # fixed start state
            joint_q = self.start_joint_q.clone()
            joint_qd = self.start_joint_qd.clone()

            with torch.no_grad():
                if self.stochastic_init:
                    (
                        joint_q[env_ids],
                        joint_qd[env_ids],
                    ) = self.get_stochastic_init(env_ids, joint_q, joint_qd)

            # assign model joint_q/qd from start pos/randomized pos
            # this effects body_q when eval_fk is called later
            self.joint_q, self.joint_qd = joint_q.view(-1), joint_qd.view(-1)
            self.model.joint_q.assign(wp.from_torch(self.joint_q.flatten()))
            self.model.joint_qd.assign(wp.from_torch(self.joint_qd.flatten()))

            # requires_grad is properly set in clear_grad()
            self.model.joint_act.zero_()
            self.state_0.clear_forces()

            # only check body state requires_grad, assumes rest of state is correct
            assert self.state_0.body_q.requires_grad == self.requires_grad
            assert self.state_0.body_qd.requires_grad == self.requires_grad
            assert self.state_0.body_f.requires_grad == self.requires_grad

            # updates state body positions after reset
            wp.sim.eval_fk(
                self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0
            )
            self.body_q = wp.to_torch(self.state_0.body_q).requires_grad_(
                self.requires_grad
            )
            self.body_qd = wp.to_torch(self.state_0.body_qd).requires_grad_(
                self.requires_grad
            )
            # reset progress buffer (i.e. episode done flag)
            self.progress_buf[env_ids] = 0
            self.num_frames = 0
            self.calculateObservations()

        return self.obs_buf

    def clear_grad(self, checkpoint=None):
        """
        cut off the gradient from the current state to previous states
        """
        if checkpoint is not None:
            self.load_checkpoint(checkpoint_data=checkpoint)
        with torch.no_grad():
            current_joint_q = self.joint_q.detach().clone()
            current_joint_qd = self.joint_qd.detach().clone()
            current_joint_act = wp.to_torch(self.model.joint_act).detach().clone()
            # grads will not be assigned since variables are detached
            self.model.joint_q.assign(wp.from_torch(current_joint_q))
            self.model.joint_qd.assign(wp.from_torch(current_joint_qd))
            self.model.joint_act.assign(wp.from_torch(current_joint_act))
            if self.model.joint_q.grad is not None:
                self.model.joint_q.grad.zero_()
                self.model.joint_qd.grad.zero_()
                self.model.joint_act.grad.zero_()
            if self.state_0.body_q.grad is not None:
                self.state_0.body_q.grad.zero_()
                self.state_0.body_qd.grad.zero_()
                self.state_0.body_f.grad.zero_()
        if self.requires_grad:
            assert self.model.joint_q.requires_grad, "joint_q requires_grad not set"
            assert self.model.joint_qd.requires_grad, "joint_qd requires_grad not set"
            assert self.model.joint_act.requires_grad, "joint_act requires_grad not set"

    def step(self, act):
        """
        Step the simulation forward by one timestep
        """
        raise NotImplementedError

    def render(self, mode="human"):
        if self.visualize:
            img = None
            if self.render_mode is RenderMode.USD:
                self.render_time += self.dt
                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.state_0)
                self.renderer.end_frame()
                if self.num_frames % self.render_freq == 0:
                    self.renderer.save()
            elif self.render_mode is RenderMode.TINY:
                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.state_0)
                if mode == "rgb_array" or self.renderer.mode == 'rgb':
                    img = self.renderer.end_frame()
                else:
                    if self.renderer.mode == 'video':
                        video_path = f"outputs/{self.stage_path}.mp4"
                        self.renderer.write_to_video(video_path)
                    self.renderer.end_frame()
                return img if img is None else np.asarray(img)

    def get_checkpoint(self, save_path=None):
        checkpoint = {}
        self.update_joints()
        joint_q, joint_qd = self.joint_q, self.joint_qd
        checkpoint["joint_q"] = joint_q.clone()
        checkpoint["joint_qd"] = joint_qd.clone()
        checkpoint["body_q"] = self.body_q.clone()
        checkpoint["body_qd"] = self.body_qd.clone()
        checkpoint["actions"] = self.actions.clone()
        checkpoint["progress_buf"] = self.progress_buf.clone()
        checkpoint["obs_buf"] = self.obs_buf.clone()
        if save_path:
            print("saving checkpoint to", save_path)
            torch.save(checkpoint, save_path)
        return checkpoint

    def load_checkpoint(self, checkpoint_data={}, ckpt_path=None):
        if ckpt_path is not None:
            checkpoint_data = torch.load(ckpt_path)
        print("loading checkpoint")
        joint_q = checkpoint_data["joint_q"].clone().view(-1, self.num_joint_q)
        joint_qd = checkpoint_data["joint_qd"].clone().view(-1, self.num_joint_q)
        self.joint_q = joint_q[: self.num_envs].flatten()
        self.joint_qd = joint_qd[: self.num_envs].flatten()
        self._joint_q.assign(to_numpy(self.joint_q))
        self._joint_qd.assign(to_numpy(self.joint_qd))
        self.body_q = checkpoint_data["body_q"].clone()[: self.state_0.body_q.size]
        self.body_qd = checkpoint_data["body_qd"].clone()[: self.state_0.body_qd.size]
        with torch.no_grad():
            # assumes self.num_envs <= number of actors in checkpoint
            self.actions[:] = checkpoint_data["actions"].clone()[: self.num_envs]
        self.progress_buf = checkpoint_data["progress_buf"].clone()[: self.num_envs]
        self.num_frames = self.progress_buf[0].item()

    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf
