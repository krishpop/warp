# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Hopper environment
#
# Shows how to set up a simulation of a rigid-body Hopper articulation based on
# the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import os
import math

import warp as wp
import warp.sim

from environment import Environment, run_env


class HopperEnvironment(Environment):
    sim_name = "env_hopper"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    use_graph_capture = True
    use_tiled_rendering = False
    show_joints = True

    def create_articulation(self, builder):
        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "../assets/hopper.xml"),
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
        )
        # initial planar velocity
        builder.joint_qd[0] = 2.0
        builder.joint_qd[1] = 2.0
        builder.joint_q[2] = 0.4
        # builder.joint_q[3] = 0.5
        builder.joint_q[5] = 0.4


if __name__ == "__main__":
    run_env(HopperEnvironment)
