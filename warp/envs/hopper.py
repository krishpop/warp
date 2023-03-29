import math
import os
import sys

import torch

# from numpy.lib.function_base import angle
from .warp_env import WarpEnv
from warp.envs.environment import Environment, RenderMode, IntegratorType

import warp as wp
import numpy as np
import random
np.set_printoptions(precision=5, linewidth=256, suppress=True)
from . import torch_utils as tu


class HopperEnv(WarpEnv):
    render_mode: RenderMode = RenderMode.USD
    integrator_type: IntegratorType = IntegratorType.EULER
    tiny_render_settings = dict(scaling=30., mode='rgb')

    def __init__(self, render=False, device='cuda:0', num_envs=4096, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, stage_path=None, env_name="HopperEnv"):
        num_obs = 11
        num_act = 3

        self.num_joint_q = 6
        self.num_joint_qd = 6

        super().__init__(num_envs, num_obs, num_act, episode_length, seed, no_grad,
                render, stochastic_init, device,
                render_mode=self.render_mode, env_name=env_name, stage_path=stage_path,)
        self.init_sim()
        self.start_joint_q[:, :3] *= 0.
        self.start_pos = self.start_joint_q[:, :2]
        self.start_rotation = self.start_joint_q[:, 2:3]
        # other parameters
        self.termination_height = -0.45
        self.termination_angle = np.pi / 6.
        self.termination_height_tolerance = 0.15
        self.termination_angle_tolerance = 0.05
        self.height_rew_scale = 1.0
        self.action_strength = 200.0
        self.action_penalty = -1e-1

        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()

        # initialize other data to be used later

    def create_articulation(self, builder):
        start_height = 0.0

        examples_dir = os.path.split(os.path.dirname(wp.__file__))[0] + "/examples"
        wp.sim.parse_mjcf(
            os.path.join(examples_dir, "assets/hopper.xml"), builder,
            density=1000.0,
            stiffness=0.0,
            damping=0.2,
            contact_ke=2.e+4,
            contact_kd=1.e+3,
            contact_kf=1.e+3,
            contact_mu=0.9,
            limit_ke=1.e+3,
            limit_kd=1.e+1,
            armature=0.1)  # TODO: add enable_self_collisions?

        # set joint targets to rest pose in mjcf
        builder.joint_q[self.num_joint_q + 3:self.num_joint_q + 6] = [0., 0., 0.]
        builder.joint_target[self.num_joint_q + 3:self.num_joint_q + 6] = [0., 0., 0., 0.]

    def assign_actions(self, actions):
        actions = actions.view((self.num_envs, self.num_actions))
        actions = torch.clip(actions, -1.0, 1.0)
        acts_per_env = int(self.model.joint_act.shape[0] / self.num_envs)
        joint_act = torch.zeros((self.num_envs*acts_per_env), dtype=torch.float32, device=self.device)
        act_types = {1: [True], 3: [], 4: [False] * 6, 5: [True] * 3, 6: [True] * 2,
                     8: [False] * 3}
        joint_types = self.model.joint_type.numpy()
        act_idx = np.concatenate([act_types[i] for i in joint_types])
        joint_act[act_idx] = actions.flatten()
        self.model.joint_act.assign(joint_act.detach().cpu().numpy())
        self.actions = actions.clone()

    def step(self, actions):
        self.assign_actions(actions)

        with wp.ScopedTimer("simulate", active=False, detailed=False):
            self.warp_step()  # iterates num_frames
            self.sim_time += self.sim_dt
        self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        if not self.requires_grad:
            self.calculateReward()

        self.reset_buf = self.reset_buf | (self.progress_buf >= self.episode_length)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.requires_grad:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                "obs_before_reset": self.obs_buf_before_reset,
                "episode_end": self.termination_buf,
            }

        if len(env_ids) > 0:
            self.reset(env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        joint_q[env_ids, 0:2] = joint_q[env_ids, 0:2] + 0.05 * (torch.rand(size=(len(env_ids), 2), device=self.device) - 0.5) * 2.
        joint_q[env_ids, 2] = (torch.rand(len(env_ids), device = self.device) - 0.5) * 0.1
        joint_q[env_ids, 3:] = joint_q[env_ids, 3:] + 0.05 * (torch.rand(size=(len(env_ids), self.num_joint_q - 3), device = self.device) - 0.5) * 2.
        joint_qd[env_ids, :] = 0.05 * (torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5) * 2.
        return joint_q[env_ids, :], joint_qd[env_ids, :]

    def reset(self, env_ids=None, force_reset=True):
        super().reset(env_ids, force_reset)  # resets state_0 and joint_q
        self.update_joints()
        self.initialize_trajectory()  # sets goal_trajectory & obs_buf
        if self.requires_grad:
            self.model.allocate_rigid_contacts(requires_grad=self.requires_grad)
            wp.sim.collide(self.model, self.state_0)
        return self.obs_buf

    def calculateObservations(self):
        self.obs_buf = torch.cat([
            self.joint_q.view(self.num_envs, -1)[:, 1:],
            self.joint_qd.view(self.num_envs, -1)], dim = -1)

    def calculateReward(self):
        height_diff = self.obs_buf[:, 0] - (self.termination_height + self.termination_height_tolerance)
        height_reward = torch.clip(height_diff, -1.0, 0.3)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)

        angle_reward = 1. * (-self.obs_buf[:, 1] ** 2 / (self.termination_angle ** 2) + 1.)

        progress_reward = self.obs_buf[:, 5]

        self.rew_buf = progress_reward + height_reward + angle_reward + torch.sum(self.actions ** 2, dim = -1) * self.action_penalty

        # reset agents
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.obs_buf[:, 0] < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)
