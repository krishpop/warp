# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Hollow Mesh
#
# Shows how to simulate hollow meshes, such as a collection of surface
# patches that fall on the ground and form hemispheres.
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim

from env.environment import Environment, run_env, RenderMode

class Demo(Environment):
    sim_name = "example_sim_hollow_mesh"
    env_offset=(2.0, 0.0, 2.0)
    nano_render_settings = dict(scaling=10.0, move_camera_target_to_center=False)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 64
    sim_substeps_xpbd = 15

    xpbd_settings = dict(
        iterations=8,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
        enable_restitution=True,
    )

    num_envs = 1

    # render_mode = RenderMode.USD

    def create_articulation(self, builder):

        self.num_bodies = 50
        self.scale = 0.8
        self.ke = 1.e+5
        self.kd = 250.0
        self.kf = 500.0

        surface = self.load_mesh(os.path.join(os.path.dirname(__file__), "assets/surfpatch.obj"))
        # mesh_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # mesh_indices = np.array([0, 1, 2], dtype=np.int32)
        # surface = wp.sim.Mesh(mesh_points, mesh_indices)
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform(
                (np.sin(i*0.5*self.scale), 1.0 + i*0.7*self.scale, np.cos(i*0.5*self.scale)*0.5),
                wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.2*i)))

            s = builder.add_shape_mesh(
                    body=b,
                    mesh=surface,
                    pos=(0.0, 0.0, 0.0),
                    scale=(self.scale, self.scale, self.scale),
                    ke=self.ke,
                    kd=self.kd,
                    kf=self.kf,
                    density=1e3,
                    is_solid=False,
                    thickness=0.1,
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


if __name__ == "__main__":
    run_env(Demo)
