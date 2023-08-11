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

from environment import Environment, run_env


@wp.kernel
def ant_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    cost: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()

    pos_base = wp.transform_get_translation(body_q[env_id * 9])
    vel_base = body_qd[env_id * 9]

    cost[env_id] = cost[env_id] + (vel_base[4]) + pos_base[0] * 10.0
    # cost[env_id] = 0.95 * cost[env_id] + 10.0 * (cart_cost + pole_cost) + 0.02 * vel_cost



class AntEnvironment(Environment):
    sim_name = "env_ant"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    use_graph_capture = True
    use_tiled_rendering = False
    show_joints = False

    controllable_dofs = [6, 7, 8, 9, 10, 11, 12, 13]
    control_gains = [100.0] * 8
    control_limits = [(-1.0, 1.0)] * 8

    def create_articulation(self, builder):
        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "../assets/nv_ant.xml"),
            builder,
            stiffness=0.0,
            damping=1.0,
            armature=0.1,
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
        builder.joint_q[:7] = [0.0, 0.7, 0.0, *wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5)]

    def evaluate_cost(self, state: wp.sim.State, cost: wp.array):
        wp.launch(
            ant_cost,
            dim=self.num_envs,
            inputs=[state.body_q, state.body_qd],
            outputs=[cost],
            device=self.device
        )

if __name__ == "__main__":
    run_env(AntEnvironment)
