# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Ant environment
#
# Shows how to set up a simulation of a rigid-body Ant articulation based on
# the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import os
import math

import warp as wp
import warp.sim

from environment import Environment, run_env, IntegratorType


@wp.kernel
def ant_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    cost: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    base_tf = body_q[env_id * 9]

    pos_base = wp.transform_get_translation(base_tf)
    rot_base = wp.transform_get_rotation(base_tf)
    vel_base = body_qd[env_id * 9]

    # cost[env_id] = cost[env_id] + (vel_base[4]) + pos_base[0] * 10.0
    # cost[env_id] = 0.95 * cost[env_id] + 10.0 * (cart_cost + pole_cost) + 0.02 * vel_cost

    termination_height = 0.27

    up_vec = wp.quat_rotate(rot_base, wp.vec3(0.0, 1.0, 0.0))
    heading_vec = wp.quat_rotate(rot_base, wp.vec3(1.0, 0.0, 0.0))

    # wp.printf("up_vec: [%.3f %.3f %.3f]\n", up_vec[0], up_vec[1], up_vec[2])
    # wp.printf("heading_vec: [%.3f %.3f %.3f]\n", heading_vec[0], heading_vec[1], heading_vec[2])

    up_reward = wp.length_sq(up_vec - wp.vec3(0.0, 0.0, -1.0))
    heading_reward = wp.length_sq(heading_vec - wp.vec3(1.0, 0.0, 0.0))
    height_reward = pos_base[1] - termination_height
    progress_reward = vel_base[3]  # double-check!

    reward = 10.0 * progress_reward + up_reward + heading_reward + height_reward
    cost[env_id] = cost[env_id] - reward


class AntEnvironment(Environment):
    sim_name = "env_ant"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 5
    sim_substeps_xpbd = 5

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    use_graph_capture = False  # True
    use_tiled_rendering = False
    show_joints = False

    controllable_dofs = [6, 7, 8, 9, 10, 11, 12, 13]
    control_gains = [50.0] * 8
    control_limits = [(-1.0, 1.0)] * 8

    integrator_type = IntegratorType.FEATHERSTONE

    def create_articulation(self, builder):
        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "../assets/nv_ant.xml"),
            builder,
            stiffness=0.0,
            damping=0.0,
            armature=0.01,
            contact_ke=1.0e4,
            contact_kd=1.0e2,
            contact_kf=1.0e4,
            contact_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
            up_axis="y",
        )
        builder.joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
        builder.joint_q[:7] = [0.0, 0.7, 0.0, *wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)]

    def evaluate_cost(self, state: wp.sim.State, cost: wp.array, step: int, horizon_length: int):
        wp.launch(ant_cost, dim=self.num_envs, inputs=[state.body_q, state.body_qd], outputs=[cost], device=self.device)


if __name__ == "__main__":
    run_env(AntEnvironment)
