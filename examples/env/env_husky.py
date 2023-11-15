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


class HuskyEnvironment(Environment):
    sim_name = "env_husky"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    use_graph_capture = False  # True
    use_tiled_rendering = False
    show_joints = False

    controllable_dofs = [6, 7, 8, 9, 10, 11, 12, 13]
    control_gains = [50.0] * 8
    control_limits = [(-1.0, 1.0)] * 8
    
    # integrator_type = IntegratorType.EULER
    
    num_envs = 1

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "../assets/husky/husky.urdf"),
            builder,
            floating=True,
            xform=wp.transform((0.0, 0.5, 0.0), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5)),
            # parse_visuals_as_colliders=True,
            # stiffness=0.0,
            # damping=0.0,
            # armature=0.01,
            # contact_ke=1.0e4,
            # contact_kd=1.0e2,
            # contact_kf=1.0e4,
            # contact_mu=1.0,
            # limit_ke=1.0e4,
            # limit_kd=1.0e1,
            # enable_self_collisions=False,
            # up_axis="y",
            collapse_fixed_joints=True,
        )
        print(builder.joint_count)
        for i in range(len(builder.joint_target)):
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_TARGET_VELOCITY
            if i % 2 == 0:
                builder.joint_target[i] = 5.0
            else:
                builder.joint_target[i] = 5.0
            builder.joint_target_ke[i] = 1000.0


if __name__ == "__main__":
    run_env(HuskyEnvironment)
