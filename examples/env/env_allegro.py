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

from environment import Environment, run_env, RenderMode


class AllegroEnvironment(Environment):
    sim_name = "example_sim_allegro"
    env_offset = (0.5, 0.0, 0.5)
    opengl_render_settings = dict(scaling=4.0)
    usd_render_settings = dict(scaling=200.0)
    episode_duration = 8.0

    sim_substeps_euler = 64
    sim_substeps_xpbd = 5

    num_envs = 100

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

    # render_mode = RenderMode.USD

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(
                os.path.dirname(__file__),
                "../assets/isaacgymenvs/kuka_allegro_description/allegro.urdf"),
            builder,
            xform=wp.transform(np.array((0.0, 0.3, 0.0)), wp.quat_rpy(-np.pi / 2, np.pi * 0.75, np.pi / 2)),
            floating=False,
            fixed_base_joint="rx, ry, rz",
            density=1e3,
            armature=0.01,
            stiffness=1000.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False)

        # for mesh in builder.shape_geo_src:
        #     if isinstance(mesh, wp.sim.Mesh):
        #         mesh.remesh(visualize=False)

        # ensure all joint positions are within limits
        offset = 3
        for i in range(offset, 16 + offset):
            builder.joint_q[i] = 0.5 * \
                (builder.joint_limit_lower[i] + builder.joint_limit_upper[i])
            builder.joint_target[i] = builder.joint_q[i]
            builder.joint_target_ke[i] = 5000.0
            builder.joint_target_kd[i] = 1.0

        cube_urdf_filename = os.path.join(
            os.path.dirname(__file__),
            "../assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf")
        cube_positions = [
            (-0.1, 0.5, 0.0),
            (0.0, 0.05, 0.05),
            (0.01, 0.15, 0.03),
            (0.01, 0.05, 0.13),
        ]
        for pos in cube_positions:
            wp.sim.parse_urdf(
                cube_urdf_filename,
                builder,
                xform=wp.transform(pos, wp.quat_identity()),
                floating=True,
                density=1e2,
                armature=0.0,
                stiffness=0.0,
                damping=0.0,
                shape_ke=1.e+3,
                shape_kd=1.e+2,
                shape_kf=1.e+2,
                shape_mu=0.5,
                limit_ke=1.e+4,
                limit_kd=1.e+1,
                parse_visuals_as_colliders=False)

        # builder.plot_articulation()
        builder.collapse_fixed_joints()
        # builder.plot_articulation()


if __name__ == "__main__":
    run_env(AllegroEnvironment)
