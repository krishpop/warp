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
import warp as wp
import warp.sim

from sim_demo import run_demo
from warp.envs import IntegratorType, WarpEnv, Environment, RenderMode
from warp.envs.autograd_utils import forward_ag, assign_act


class AllegroSim(Environment):
    env_name = "example_sim_allegro"
    env_offset = (6.0, 0.0, 6.0)
    tiny_render_settings = dict(scaling=15.0)
    usd_render_settings = dict(scaling=10.0)

    sim_substeps_euler = 64
    sim_substeps_xpbd = 5
    render_mode: RenderMode = RenderMode.NONE

    num_envs = 1
    target_pos = np.array((0.0, 0.4, 0.4))

    xpbd_settings = dict(
        iterations=10,
        joint_linear_relaxation=1.0,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
        enable_restitution=True,
    )

    use_graph_capture = True

    rigid_contact_margin = 0.005
    rigid_mesh_contact_max = 100
    # contact thickness to apply around mesh shapes
    contact_thickness = 0.0

    def create_articulation(self, builder):
        floating_base = False
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
            floating=floating_base,
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
        q_offset = 7 if floating_base else 0
        qd_offset = 6 if floating_base else 0
        for i in range(16):
            builder.joint_q[i + q_offset] = 0.5 * (
                builder.joint_limit_lower[i + qd_offset]
                + builder.joint_limit_upper[i + qd_offset]
            )
            builder.joint_target[i] = builder.joint_q[i + q_offset]
            builder.joint_target_ke[i] = 50000.0
            builder.joint_target_kd[i] = 10.0

        wp.sim.parse_urdf(
            os.path.join(
                os.path.dirname(__file__),
                "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf",
            ),
            builder,
            xform=wp.transform(np.array((-0.1, 0.5, 0.0)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e3,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=0.5,
            shape_thickness=self.contact_thickness,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            parse_visuals_as_colliders=False,
        )

        wp.sim.parse_urdf(
            os.path.join(
                os.path.dirname(__file__),
                "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf",
            ),
            builder,
            xform=wp.transform(np.array((0.0, 0.05, 0.05)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e3,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=0.5,
            shape_thickness=self.contact_thickness,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            parse_visuals_as_colliders=False,
        )

        wp.sim.parse_urdf(
            os.path.join(
                os.path.dirname(__file__),
                "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf",
            ),
            builder,
            xform=wp.transform(np.array((0.01, 0.15, 0.03)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e3,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=0.5,
            shape_thickness=self.contact_thickness,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            parse_visuals_as_colliders=False,
        )
        wp.sim.parse_urdf(
            os.path.join(
                os.path.dirname(__file__),
                "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf",
            ),
            builder,
            xform=wp.transform(np.array((0.01, 0.05, 0.13)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e3,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=0.5,
            shape_thickness=self.contact_thickness,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            parse_visuals_as_colliders=False,
        )

    def step(self, actions, profiler=None):
        active, detailed = False, False
        if profiler is not None:
            active, detailed = False, True

        with wp.ScopedTimer(
            "simulate", active=active, detailed=detailed, dict=profiler
        ):
            actions = torch.clip(actions, -1.0, 1.0)
            self.actions = actions.view(self.num_envs, -1)
            actions = self.action_strength * actions

            if self.requires_grad:
                # does this cut off grad to prev timestep?
                assert (
                    self.model.body_q.requires_grad
                    and self.state_0.body_q.requires_grad
                )
                # all grads should be from joint_q, not from body_q
                # with torch.no_grad():
                body_q, body_qd = (
                    self.body_q.detach().clone(),
                    self.body_qd.detach().clone(),
                )

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
                        self.state_1,
                        self.state_0,
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
            with wp.ScopedTimer(
                "reset", active=active, detailed=detailed, dict=profiler
            ):
                self.reset(env_ids)

        if self.visualize and self.render_mode is not RenderMode.TINY:
            with wp.ScopedTimer(
                "render", active=active, detailed=detailed, dict=profiler
            ):
                self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extra

    def calculateObservations(self):
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(
            self.num_envs, -1
        )
        body_q = self.body_q.view(self.num_envs, -1, 7)
        body_qd = self.body_qd.view(self.num_envs, -1, 6)
        x = body_q[:, 0:1, 0] - body_q[:, 1:2, 0]  # joint_q[:, 0:1]
        xdot = body_qd[:, 1, 3:4]  # joint_qd[:, 0:1]

        theta = joint_q[:, 1:2]
        theta_dot = joint_qd[:, 1:2]

        # observations: [x, xdot, sin(theta), cos(theta), theta_dot]
        self.obs_buf = torch.cat(
            [x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1
        )

    def calculateReward(self):
        """Computes distance from hand to target and sets reward accordingly"""
        # reward is negative distance to target
        body_q = self.body_q.view(self.num_envs, -1, 7)
        x = body_q[:, 0, :3] - torch.as_tensor(
            self.target_pos, dtype=torch.float32, device=self.device
        )
        self.rew_buf = -torch.norm(x, dim=-1)


if __name__ == "__main__":
    run_demo(AllegroSim)
    # import argparse
    # parser.add_argument(
    #         "--num_envs",
    #         help="Number of environments to simulate",
    #         type=int,
    #     )
    # args = parser.parse_args()
    # num_envs = args.num_envs
    # env = AllegroSim(num_envs=num_envs, requires_grad=True)
    # parser = argparse.ArgumentParser()
    #
    #     parser.add_argument(
    #         "--profile", help="Enable profiling", type=bool, default=profile
    #     )
    # return env.run()
    # env.reset()
    # for _ in range(100):
    #     obs, rew, done, info = env.step(torch.rand(1, 7))
