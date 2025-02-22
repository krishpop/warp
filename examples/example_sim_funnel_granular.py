# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Funnel Granular
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

from env.environment import Environment, run_env, IntegratorType

class Demo(Environment):
    sim_name = "example_sim_funnel_granular"
    env_offset=(20, 0.0, 20)
    nano_render_settings = dict(scaling=0.25)
    usd_render_settings = dict(scaling=5.0)

    num_envs = 4

    sim_substeps_euler = 32
    sim_substeps_xpbd = 15

    xpbd_settings = dict(
        iterations=1,
        enable_restitution=True,
        rigid_contact_con_weighting=False,
    )

    def load_mesh(self, filename, use_meshio=True):
        if use_meshio:
            import meshio
            m = meshio.read(filename)
            mesh_points = np.array(m.points)
            mesh_indices = np.array(m.cells[0].data, dtype=np.int32).flatten()
        else:
            import openmesh
            m = openmesh.read_trimesh(filename)
            mesh_points = np.array(m.points())
            mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32).flatten()
        return wp.sim.Mesh(mesh_points, mesh_indices)

    def create_articulation(self, builder):
        self.num_bodies = 5
        self.scale = 0.5

        self.ke = 1.e+5
        self.kd = 250.0
        self.kf = 500.0
        self.mu  = 1.0
        self.restitution = 0.9

        builder.set_ground_plane(
            ke=self.ke, 
            kd=self.kd, 
            kf=self.kf,
            mu=self.mu,
            restitution=self.restitution,
        )

        # funnel
        funnel_mesh = self.load_mesh(os.path.join(os.path.dirname(__file__), f"assets/funnel2.obj"))

        funnel = builder.add_body(
            origin=wp.transform((0.0, 12.0, 0.0), (wp.sin(math.pi/4), 0.0,  0.0, wp.sin(math.pi/4)))
        )
        builder.add_shape_mesh(
            body=funnel,
            mesh=funnel_mesh,
            pos=(0.0, 0.0, 0.0),
            scale=(0.1, 0.1, 0.1),
            ke=self.ke, 
            kd=self.kd, 
            kf=self.kf,
            mu=self.mu,
            restitution=self.restitution,
            density=0.0,
            thickness=0.01,
        )

        height = 30
        width = 10
        depth = 10
        radius = 0.2
        spacing = 0.7
        # spheres
        for i in range(height):
            for j in range(width):
                for k in range(depth):
                    pos = np.array((
                        (k - (width-1)/2.0)*(spacing+radius),
                        i * (spacing+radius)+22.0,
                        (j - (depth-1)/2.0)*(spacing+radius)))
                    # add jitter
                    pos += np.random.uniform(-spacing, spacing, size=3) * 0.5
                    b = builder.add_body(origin=wp.transform(pos, wp.quat_identity()))

                    s = builder.add_shape_sphere(
                        pos=(0.0, 0.0, 0.0),
                        radius=radius,
                        body=b,
                        ke=self.ke,
                        kd=self.kd,
                        kf=self.kf,
                        mu=self.mu,
                        restitution=self.restitution)

        builder.ground = [0.0, 1.0, 0.0, 0.0]

    def before_simulate(self):
        self.model.rigid_contact_rolling_friction = 0.05
        self.model.rigid_contact_torsion_friction = 0.05
        self.model.rigid_contact_margin = 0.05


if __name__ == "__main__":
    run_env(Demo)
