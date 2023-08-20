# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Mano
#
# Shows how to set up a simulation of a rigid-body MANO hand articulation
# from a MuJoCo file using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import os

import numpy as np
import warp as wp
import warp.sim

from environment import Environment, run_env, RenderMode


class ManoEnvironment(Environment):
    sim_name = "example_sim_mano"
    env_offset = (0.5, 0.0, 0.5)
    opengl_render_settings = dict(scaling=40.0, draw_axis=False)
    usd_render_settings = dict(scaling=200.0)
    episode_duration = 8.0

    sim_substeps_euler = 64
    sim_substeps_xpbd = 8

    num_envs = 1

    # show_joints = True

    xpbd_settings = dict(
        iterations=10,
        # joint_linear_relaxation=0.1,
        # joint_angular_relaxation=0.1,
        # rigid_contact_relaxation=1.0,
        # rigid_contact_con_weighting=True,
    )

    use_tiled_rendering = False
    use_graph_capture = True

    activate_ground_plane = False

    show_joints = True

    # render_mode = RenderMode.USD

    def create_articulation(self, builder):
        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "../assets/mano_hand/auto_mano_fixed_base.xml"),
            builder,
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_rpy(-np.pi / 2, np.pi * 0.75, np.pi / 2)),
            density=1e3,
            armature=0.1,
            stiffness=10.0,
            damping=0.0,
            scale=1.0,
            # shape_ke=1.e+3,
            # shape_kd=1.e+2,
            # shape_kf=1.e+2,
            # shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=True)

        # builder.plot_articulation()
        builder.collapse_fixed_joints()
        # builder.plot_articulation()

        # ensure all joint positions are within limits
        offset = 0  # skip floating base
        for i in range(len(builder.joint_limit_lower)):
            builder.joint_q[i + offset] = 0.5 * \
                (builder.joint_limit_lower[i] + builder.joint_limit_upper[i])
            builder.joint_target[i] = builder.joint_q[i + offset]
            builder.joint_target_ke[i] = 5000.0
            builder.joint_target_kd[i] = 1.0


if __name__ == "__main__":
    run_env(ManoEnvironment)
