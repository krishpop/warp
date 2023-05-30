# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Shadow Hand
#
# Shows how to set up a simulation of a rigid-body Shadow hand articulation
# from an MJCF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import os

import numpy as np
import warp as wp
import warp.sim

from environment import Environment, run_env, RenderMode


class ShadowHandEnvironment(Environment):
    sim_name = "example_sim_shadow_hand"
    env_offset = (0.5, 0.0, 0.5)
    opengl_render_settings = dict(scaling=4.0)
    usd_render_settings = dict(scaling=200.0)
    episode_duration = 8.0

    sim_substeps_euler = 64
    sim_substeps_xpbd = 8

    num_envs = 1

    show_joints = False

    xpbd_settings = dict(
        iterations=10,
        joint_linear_relaxation=1.0,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )

    use_tiled_rendering = False
    use_graph_capture = True

    activate_ground_plane = False

    # render_mode = RenderMode.USD

    def create_articulation(self, builder):
        wp.sim.parse_mjcf(
            os.path.join(
                os.path.dirname(__file__),
                "../assets/shadow_hand/right_hand.xml"),
            builder,
            stiffness=10.0,
            ignore_classes=["plastic_collision"])

        # builder.plot_articulation()
        builder.collapse_fixed_joints()
        # builder.plot_articulation()


if __name__ == "__main__":
    run_env(ShadowHandEnvironment)
