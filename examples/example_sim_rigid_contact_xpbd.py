# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Contact
#
# Shows how to set up free rigid bodies with different shape types falling
# and colliding against the ground using wp.sim.ModelBuilder().
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

from env.environment import Environment, run_env

class Demo(Environment):
    sim_name = "example_sim_rigid_contact_xpbd"
    num_envs = 1

    tiny_render_settings = dict(scaling=5)

    def create_articulation(self, builder):

        self.num_bodies = 3
        self.scale = 1.5
        self.ke = 1.e+5
        self.kd = 250.0
        self.kf = 500.0
        restitution = 1.0

        # boxes
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((i, 2.2 + 0.3*i, 0.0), wp.quat_identity()))
            # b = builder.add_body(origin=wp.transform((i, 0.6, 0.0), wp.quat_rpy(0.2, 0.5, 0.8)))

            s = builder.add_shape_box( 
                pos=(0.0, 0.0, 0.0),
                hx=0.25*self.scale,
                hy=0.12*self.scale,
                hz=0.12*self.scale,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.05,
                restitution=restitution)

        # spheres
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((0.01 * i + 1.0, 1.3*self.scale * i + 2.5, 1.0), wp.quat_identity()))

            s = builder.add_shape_sphere(
                pos=(0.0, 0.0, 0.0),
                radius=0.25*self.scale,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=1.0,
                restitution=restitution)

        # capsules
        for i in range(self.num_bodies):
            
            # b = builder.add_body(origin=wp.transform((0.0, 2.0, 4.0), wp.quat_rpy(0.1, 0.0, 0.3)))
            b = builder.add_body(origin=wp.transform((0.0, 2.0 + 0.8*i, 2.0), wp.quat_identity()))

            s = builder.add_shape_capsule( 
                pos=(0.0, 0.0, 0.0),
                radius=0.25*self.scale,
                half_height=self.scale*0.5,
                up_axis=0,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.01,
                restitution=restitution)

        if True:
            builder.add_shape_plane(
                plane=(0.3, 1.0, 0.0, 1.5),
                width=1.5,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.0,
                restitution=restitution)

        # add ground plane
        builder.add_shape_plane(
            plane=(0.0, 1.0, 0.0, 0.0),
            width=0.0, length=0.0,
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
            mu=0.0,
            restitution=restitution)
            
        # initial spin 
        # for i in range(len(builder.body_qd)):
        #     # builder.body_qd[i] = (0.0, 2.0, 10.0, -1.5, 0.0, 0.0)
        #     builder.body_qd[i] = (10.0, 0.0, 0.0, 0.3, 2.5, -4.0)


if __name__ == "__main__":
    run_env(Demo)
