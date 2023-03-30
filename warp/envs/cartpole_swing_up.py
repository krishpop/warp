import torch
import warp as wp
import os
import numpy as np
import math

from .warp_env import WarpEnv
from .environment import RenderMode, IntegratorType
from . import torch_utils as tu
from .autograd_utils import IntegratorSimulate, assign_act



class CartPoleSwingUpEnv(WarpEnv):
    render_mode: RenderMode = RenderMode.TINY
    integrator_type: IntegratorType = IntegratorType.XPBD
    activate_ground_plane: bool = False

    def __init__(
        self,
        render=False,
        device="cuda",
        num_envs=1024,
        seed=0,
        episode_length=240,
        no_grad=True,
        stochastic_init=False,
        early_termination=False,
        stage_path=None,
        env_name="CartPoleSwingUpEnv"
    ):

        num_obs = 5
        num_act = 1

        super().__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            seed,
            no_grad,
            render,
            stochastic_init,
            device,
            env_name=env_name,
            render_mode=self.render_mode,
            stage_path=stage_path
        )
        self.num_joint_q = 2
        self.num_joint_qd = 2
        self.early_termination = early_termination
        self.init_sim()
        dof_count = int(self.model.joint_act.shape[0] / self.num_envs)
        act = wp.zeros(self.num_envs * self.num_acts, dtype=self.model.joint_act.dtype,
                       device=self.device)
        assert dof_count * self.num_envs == self.model.joint_act.size
        self.simulate_params = {"model": self.model, "integrator": self.integrator,
                                "dt": self.sim_dt, "substeps": self.sim_substeps,
                                "state_in": self.state_0, "state_out": self.state_1}
        self.act_params = {"q_offset": 0, "joint_act": self.model.joint_act,
                           "act": act, "num_envs": self.num_envs,
                           "dof_count": dof_count, "num_acts": self.num_acts,
                           }
        self.graph_capture_params = {"capture_graph": self.use_graph_capture}
        self.graph_capture_params["joint_q_end"] = wp.zeros_like(self.model.joint_q)
        self.graph_capture_params["joint_qd_end"] = wp.zeros_like(self.model.joint_qd)
        if self.use_graph_capture and self.requires_grad:
            backward_model = self.model if not self.use_graph_capture else self.builder.finalize()
            self.graph_capture_params["bwd_model"] = backward_model
            self.graph_capture_params["bwd_joint_q_end"] = wp.zeros_like(backward_model.joint_q)
            self.graph_capture_params["bwd_joint_qd_end"] = wp.zeros_like(backward_model.joint_qd)
            self.graph_capture_params["tape"] = wp.Tape()
            self.graph_capture_params["bw_tape"] = wp.Tape()
            self.simulate_params["state_list"] = [backward_model.state(requires_grad=True) for _ in range(self.sim_substeps + 1)]

        # action parameters
        self.action_strength = 1000.0

        # loss related
        self.pole_angle_penalty = 1.0
        self.pole_velocity_penalty = 0.1

        self.cart_position_penalty = 0.05
        self.cart_velocity_penalty = 0.1

        self.cart_action_penalty = 0.0

    def create_articulation(self, builder):
        examples_dir = os.path.split(os.path.dirname(wp.__file__))[0] + "/examples"
        wp.sim.parse_urdf(
            os.path.join(examples_dir, "assets/cartpole.urdf"),
            builder,
            xform=wp.transform(np.zeros(3), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
            floating=False,
            density=0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False)

    def assign_actions(self, actions):
        actions = actions.flatten() * self.action_strength
        # actions = torch.clip(actions, -1.0, 1.0)
        self.act_params['act'].assign(wp.from_torch(actions))
        assign_act(**self.act_params)

    def step(self, actions):
        with wp.ScopedTimer("simulate", active=False, detailed=False):
            actions = torch.clip(actions, -1.0, 1.0)
            self.actions = actions.view(self.num_envs, -1)
            actions = self.action_strength * actions

            if self.requires_grad:
                # does this cut off grad to prev timestep?
                body_q = wp.to_torch(self.state_0.body_q)
                body_qd = wp.to_torch(self.state_0.body_qd)
                body_q.requires_grad = self.requires_grad
                body_qd.requires_grad = self.requires_grad
                assert (
                    self.model.body_q.requires_grad
                    and self.state_0.body_q.requires_grad
                )
                state_out = self.model.state(requires_grad=True)
                self.joint_q, self.joint_qd, self.state_0 = IntegratorSimulate.apply(
                    self.simulate_params,
                    self.graph_capture_params,
                    self.act_params,
                    action.flatten(),
                    body_q,
                    body_qd,
                )
            else:
                self.assign_actions(self.actions)
                self.warp_step()

            self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        if self.requires_grad:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                "obs_before_reset": self.obs_buf_before_reset,
                "episode_end": self.termination_buf,
            }

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        with wp.ScopedTimer("reset", active=False, detailed=False):
            if len(env_ids) > 0:
                self.reset(env_ids)

        with wp.ScopedTimer("render", active=False, detailed=False):
            self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        rand_init_q = np.pi * (
            torch.rand(size=(len(env_ids), self.num_joint_q), device=self.device) - 0.5
        )
        rand_init_qd = 0.5 * (
            torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5
        )
        return joint_q[env_ids] + rand_init_q, joint_qd[env_ids] + rand_init_qd

    def calculateObservations(self):
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(
            self.num_envs, -1
        )

        x = joint_q[:, 0:1]
        theta = joint_q[:, 1:2]
        xdot = joint_qd[:, 0:1]
        theta_dot = joint_qd[:, 1:2]

        # observations: [x, xdot, sin(theta), cos(theta), theta_dot]
        self.obs_buf = torch.cat(
            [x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1
        )

    def calculateReward(self):
        joint_q = self.joint_q.view(self.num_envs, -1)
        joint_qd = self.joint_qd.view(self.num_envs, -1)

        x = joint_q[:, 0]
        theta = tu.normalize_angle(joint_q[:, 1])
        xdot = joint_qd[:, 0]
        theta_dot = joint_qd[:, 1]

        self.rew_buf = (
            -torch.pow(theta, 2.0) * self.pole_angle_penalty
            - torch.pow(theta_dot, 2.0) * self.pole_velocity_penalty
            - torch.pow(x, 2.0) * self.cart_position_penalty
            - torch.pow(xdot, 2.0) * self.cart_velocity_penalty
            - torch.sum(self.actions**2, dim=-1) * self.cart_action_penalty
        )

        # reset agents
        self.reset_buf = torch.where(
            self.progress_buf > self.episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
