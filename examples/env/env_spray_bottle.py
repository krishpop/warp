# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Spray bottle environment
#
# Shows how to load a URDF of a spray bottle.
#
###########################################################################

import os
import warp as wp
import warp.sim

from environment import Environment, run_env


class SprayBottleEnvironment(Environment):
    sim_name = "env_spray_bottle"
    env_offset = (0.5, 0.0, 0.5)
    opengl_render_settings = dict(scaling=5.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    activate_ground_plane = True
    use_graph_capture = False

    num_envs = 100

    show_joints = False

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "../assets/spray_bottle/mobility.urdf"),
            builder,
            xform=wp.transform((0.0, 0.22, 0.0)),
            floating=True,
            density=1000.0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False)

        # builder.joint_act[5] = 0.15


if __name__ == "__main__":
    run_env(SprayBottleEnvironment)
