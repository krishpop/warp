# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim xarm
#
# Shows how to set up a simulation of a rigid-body xarm articulation 
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim

from environment import Environment, run_env


class XarmEnvironment(Environment):
    sim_name = "example_sim_xarm"
    env_offset=(0.8, 0.0, 0.8)
    nano_render_settings = dict(scaling=10.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    xpbd_settings = dict(
        iterations=2,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.5,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )

    activate_ground_plane = False

    def create_articulation(self, builder):
            wp.sim.parse_urdf(os.path.join(os.path.dirname(__file__), "../assets/xarm/xarm6_robot.urdf"), builder,
                xform=wp.transform((0.0, 0.0, 0.0), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
                floating=False, 
                density=0.0,
                armature=0.01,
                stiffness=0.0,
                damping=0.0,
                shape_ke=1.e+4,
                shape_kd=1.e+2,
                shape_kf=1.e+2,
                shape_mu=1.0,
                limit_ke=1.e+4,
                limit_kd=1.e+1)

            # joint initial positions
            builder.joint_q[-3:] = [0.0, 0.3, 0.0]
            builder.joint_target[:3] = [0.0, 0.0, 0.0]


if __name__ == "__main__":
    run_env(XarmEnvironment)
