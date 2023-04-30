# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Allegro
#
# Shows how to set up a simulation of a rigid-body Allegro hand articulation
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import os

import numpy as np

# from sim_demo import run_demo
import torch
import warp as wp
from tqdm import trange
from warp.envs import Environment, IntegratorType, RenderMode, WarpEnv
from warp.envs.autograd_utils import assign_act, forward_ag
import warp.sim


class AllegroHandEnv(WarpEnv):
    integrator_type: IntegratorType = IntegratorType.XPBD
    activate_ground_plane: bool = False
    use_graph_capture: bool = False
    env_offset = (6.0, 0.0, 6.0)
    tiny_render_settings = dict(scaling=15.0)
    usd_render_settings = dict(scaling=10.0)

    sim_substeps_euler = 64
    sim_substeps_xpbd = 5
    render_mode: RenderMode = RenderMode.USD

    target_pos = np.array((0.0, 0.4, 0.4))

    xpbd_settings = dict(
        iterations=10,
        joint_linear_relaxation=1.0,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
        enable_restitution=True,
    )

    rigid_contact_margin = 0.005
    rigid_mesh_contact_max = 100
    # contact thickness to apply around mesh shapes
    contact_thickness = 0.0
    action_strength = 0.05

    def __init__(
        self,
        render=False,
        device="cuda",
        num_envs=256,
        seed=0,
        episode_length=240,
        no_grad=True,
        stochastic_init=False,
        early_termination=False,
        stage_path=None,
        env_name="AllegroHandEnv",
        floating_base=False,
        graph_capture=False,
    ):
        num_obs = 16 * 2
        num_act = 16
        self.use_graph_capture = graph_capture
        self.floating_base = floating_base
        if floating_base:
            num_obs += 7 * 2 + 6 * 2
            num_act += 6

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
            stage_path=stage_path,
        )
        self.num_joint_q = 16
        self.num_joint_qd = 16
        self.early_termination = early_termination
        self.init_sim()

        # create mappings for joint and body indices
        self.body_name_to_idx, self.joint_name_to_idx = {}, {}
        for i, body_name in enumerate(self.model.body_name):
            body_ind = self.body_name_to_idx.get(body_name, [])
            body_ind.append(i)
            self.body_name_to_idx[body_name] = body_ind

        for i, joint_name in enumerate(self.model.joint_name):
            joint_ind = self.joint_name_to_idx.get(joint_name, [])
            joint_ind.append(i)
            self.joint_name_to_idx[joint_name] = joint_ind

        self.body_name_to_idx = {k: np.array(v) for k, v in self.body_name_to_idx.items()}
        self.joint_name_to_idx = {k: np.array(v) for k, v in self.joint_name_to_idx.items()}

        self.setup_autograd_vars()
        self.prev_contact_count = np.zeros(self.num_envs, dtype=int)
        self.contact_count_changed = torch.zeros_like(self.reset_buf)
        self.contact_count = wp.clone(self.model.rigid_contact_count)

    def create_articulation(self, builder):
        xform = wp.transform(
            np.array((0.0, 0.3, 0.0)),
            wp.quat_rpy(-np.pi / 2, np.pi * 0.75, np.pi / 2),
        )
        wp.sim.parse_urdf(
            os.path.join(
                os.path.dirname(__file__),
                "assets/isaacgymenvs/kuka_allegro_description/allegro.urdf",
            ),
            builder,
            xform=xform,
            floating=self.floating_base,
            density=1e3,
            armature=0.01,
            stiffness=1000.0,
            damping=0.0,
            shape_ke=1.0e3,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=0.5,
            shape_thickness=self.contact_thickness,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
        )

        # ensure all joint positions are within limits
        q_offset = 7 if self.floating_base else 0
        qd_offset = 6 if self.floating_base else 0
        for i in range(16):
            builder.joint_q[i + q_offset] = 0.5 * (
                builder.joint_limit_lower[i + qd_offset] + builder.joint_limit_upper[i + qd_offset]
            )
            builder.joint_target[i] = builder.joint_q[i + q_offset]
            builder.joint_target_ke[i] = 50000.0
            builder.joint_target_kd[i] = 10.0

    def assign_actions(self, actions):
        actions = actions.flatten() * self.action_strength
        # actions = torch.clip(actions, -1.0, 1.0)
        self.act_params["act"].assign(wp.from_torch(actions))
        assign_act(**self.act_params)

    def step(self, actions, profiler=None):
        active, detailed = False, False
        if profiler is not None:
            active, detailed = False, True

        with wp.ScopedTimer("simulate", active=active, detailed=detailed, dict=profiler):
            actions = torch.clip(actions, -1.0, 1.0)
            self.actions = actions.view(self.num_envs, -1)
            actions = self.action_strength * actions

            if self.requires_grad:
                # does this cut off grad to prev timestep?
                assert self.model.body_q.requires_grad and self.state_0.body_q.requires_grad
                # all grads should be from joint_q, not from body_q
                body_q = self.body_q.clone()
                body_qd = self.body_qd.clone()

                ret = forward_ag(
                    self.simulate_params,
                    self.graph_capture_params,
                    self.act_params,
                    actions.flatten(),
                    body_q,
                    body_qd,
                )
                # swap states so start from correct next state
                if not self.use_graph_capture:
                    (
                        self.simulate_params["state_in"],
                        self.simulate_params["state_out"],
                    ) = (
                        self.simulate_params["state_out"],
                        self.simulate_params["state_in"],
                    )
                self.joint_q, self.joint_qd, self.body_q, self.body_qd = ret

            else:
                self.assign_actions(self.actions)
                self.update()

            self.sim_time += self.sim_dt * self.sim_substeps

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

        if len(env_ids) > 0:
            with wp.ScopedTimer("reset", active=active, detailed=detailed, dict=profiler):
                self.reset(env_ids)

        if self.visualize and self.render_mode is not RenderMode.TINY:
            with wp.ScopedTimer("render", active=active, detailed=detailed, dict=profiler):
                self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def calculateObservations(self):
        """Computes observations from current state"""
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(self.num_envs, -1)
        body_q = self.body_q.view(self.num_envs, -1, 7)[:, 0, :3]  # pos
        body_qd = self.body_qd.view(self.num_envs, -1, 6)[..., 0, 4:5]  # ang vel

        self.obs_buf = torch.cat([body_q, body_qd, joint_q, joint_qd], dim=-1)

    def calculateReward(self):
        """Computes distance from hand to target and sets reward accordingly"""
        # reward is negative distance to target
        body_q = self.body_q.view(-1, 7)[self.body_name_to_idx["index_link_1"], :3].view(self.num_envs, -1)
        x = body_q - torch.as_tensor(self.target_pos, dtype=torch.float32, device=self.device).view(1, -1)
        self.rew_buf = -torch.norm(x, dim=-1)


if __name__ == "__main__":
    # run_demo(AllegroSim)
    import time

    requires_grad = True
    env = AllegroHandEnv(num_envs=256, no_grad=(not requires_grad), render=True)
    next_act = torch.randn(
        env.act_params["act"].shape,
        dtype=torch.float32,
        device=env.device,
        requires_grad=requires_grad,
    )
    env.reset()
    start_time = time.perf_counter()
    for i in trange(env.episode_length):
        act = next_act
        obs, rew, done, info = env.step(act)
        next_act = torch.randn(
            env.act_params["act"].shape, dtype=torch.float32, device=env.device, requires_grad=requires_grad
        )
        if env.requires_grad:
            rew.sum().backward()
            act = act + act.grad * 0
    end_time = time.perf_counter()
    time_lapse = end_time - start_time
    print("steps/sec:", env.num_envs * env.episode_length / time_lapse)
