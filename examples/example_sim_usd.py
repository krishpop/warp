# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Ant
#
# Shows how to set up a simulation of a rigid-body Ant articulation based on
# the OpenAI gym environment using the wp.sim.ModelBuilder() and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import os
import math

import numpy as np

import warp as wp


# wp.config.verify_cuda = True
# wp.config.verify_fp = True

import warp as wp
import warp.sim

from environment import Environment, run_env, IntegratorType, RenderMode


from tqdm import trange

# wp.init()
# wp.set_device("cpu")

class Demo(Environment):
    sim_name = "example_sim_usd"
    env_offset=(6.0, 0.0, 6.0)
    tiny_render_settings = dict(scaling=15.0)
    usd_render_settings = dict(scaling=200.0)

    sim_substeps_euler = 64
    sim_substeps_xpbd = 8

    xpbd_settings = dict(
        iterations=10,
        enable_restitution=True,
        joint_linear_relaxation=0.8,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )

    activate_ground_plane = False
    num_envs = 1

    render_mode = RenderMode.TINY
    # integrator_type = IntegratorType.EULER

    plot_body_coords = True

    def create_articulation(self, builder):

        folder = os.path.join(os.path.dirname(__file__), "assets", "usd_physics")
        settings = wp.sim.parse_usd(
            # os.path.join(folder, "box_on_quad.usd"),
            # os.path.join(folder, "contact_pair_filtering.usd"),
            os.path.join(folder, "distance_joint.usd"),
            # os.path.join(folder, "prismatic_joint.usda"),
            # os.path.join(folder, "revolute_joint.usd"),
            # os.path.join(folder, "d6_joint.usda"),
            # os.path.join(folder, "spheres_with_materials.usd"),
            # os.path.join(folder, "material_density.usda"),
            # os.path.join(folder, "shapes_on_plane.usda"),
            # os.path.join(folder, "articulation.usda"),
            # os.path.join(folder, "ropes.usda"),
            builder,
            default_thickness=0.01
        )
    
        self.frame_dt = 1.0 / settings["fps"]
        self.episode_duration = 15.0 #settings["duration"]
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.episode_frames = int(self.episode_duration/self.frame_dt)
        self.sim_steps = int(self.episode_duration / self.sim_dt)
        self.sim_time = 0.0
        self.render_time = 0.0
        self.up_axis = settings["up_axis"]

    def before_simulate(self):
        print("COM", self.model.body_com.numpy())
        print("Mass", self.model.body_mass.numpy())
        print("Inertia", self.model.body_inertia.numpy())
        print("shape_transform", self.model.shape_transform.numpy())
        print("geo_scale", self.model.shape_geo.scale.numpy())
        # print("collision filters", sorted(list(self.builder.shape_collision_filter_pairs)))
        if self.model.joint_count > 0:
            print("joint parent", self.model.joint_parent.numpy())
            print("joint child", self.model.joint_child.numpy())
            if len(self.model.joint_q) > 0:
                print("joint q", self.model.joint_q.numpy())
            if len(self.model.joint_axis) > 0:
                print("joint axis", self.model.joint_axis.numpy())
                print("joint target", self.model.joint_target.numpy())
                print("joint target ke", self.model.joint_target_ke.numpy())
                print("joint target kd", self.model.joint_target_kd.numpy())
                print("joint limit lower", self.model.joint_limit_lower.numpy())
                print("joint limit upper", self.model.joint_limit_upper.numpy())
            print("joint_X_p", self.model.joint_X_p.numpy())
            print("joint_X_c", self.model.joint_X_c.numpy())
        print("body_q", self.state.body_q.numpy())

if __name__ == "__main__":
    run_env(Demo)
