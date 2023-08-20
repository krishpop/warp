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
    use_graph_capture = True

    show_rigid_contact_points = True

    num_envs = 4

    show_joints = False

    rigid_contact_margin = 0.001

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "../assets/spray_bottle/mobility.urdf"),
            builder,
            xform=wp.transform((0.0, 0.22, 0.0), wp.quat_from_axis_angle((0.7, 0.0, 1.0), 0.1)),
            floating=True,
            density=1.0,
            armature=1e-4,
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

        print("builder.body_mass", "\n".join(map(str, builder.body_mass)))
        print("builder.body_inertia", "\n".join(map(str, builder.body_inertia)))
        print("builder.body_com", "\n".join(map(str, builder.body_com)))
        
        # for mesh in builder.shape_geo_src:
        #     if isinstance(mesh, wp.sim.Mesh):
        #         mesh.remesh(method="alphashape", visualize=False)
        #         mesh.remesh(method="ftetwild", visualize=False)
        
        # builder.plot_articulation()
        builder.collapse_fixed_joints()
        # builder.plot_articulation()


if __name__ == "__main__":
    run_env(SprayBottleEnvironment)
