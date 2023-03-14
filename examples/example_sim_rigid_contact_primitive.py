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

from env.environment import Environment, run_env, IntegratorType, RenderMode

class Demo(Environment):

    sim_name = "example_sim_contact_primitive"
    env_offset=(10.0, 0.0, 20.0)
    tiny_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=10.0)

    episode_duration = 20.0

    sim_substeps_euler = 32
    sim_substeps_xpbd = 10

    xpbd_settings = dict(
        iterations=3,
        rigid_contact_relaxation=0.8,
        enable_restitution=True,
    )

    num_envs = 1

    separate_collision_group_per_env = True
    integrator_type = IntegratorType.XPBD
    render_mode = RenderMode.USD

    use_graph_capture = True

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

        self.num_bodies = 20
        self.scale = 0.5
        self.ke = 1.e+5 
        self.kd = 250.0
        self.kf = 500.0
        self.restitution = 0.2

        scaling = np.linspace(1., 1.0, self.num_bodies)
        rot90s = [
            wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.25),
            wp.quat_from_axis_angle((0.0, 1.0, 0.0), -math.pi*0.25),
        ]
        # boxes
        for i in range(1, self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((0.0, 0.5 * i + 8.3, 1.0), rot90s[i%2]))

            s = builder.add_shape_box( 
                pos=(0.0, 0.0, 0.0),
                hx=0.5*self.scale * scaling[i],
                hy=0.2*self.scale * scaling[i],
                hz=0.2*self.scale * scaling[i],
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.5,
                density=1e3,
                restitution=self.restitution)

        if True:
            b = builder.add_body(origin=wp.transform((0.0, 22.5, 0.0), wp.quat_identity()))
            s = builder.add_shape_sphere(
                pos=(0.0, 0.0, 0.0),
                radius=0.5, 
                density=1e3,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.5,
                restitution=self.restitution)
            builder.body_qd[-1] = [0.0, 0.0, 0.0, -0.2, 0.0, 0.0]


        # spheres
        for i in range(1, self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((0.01 * i, 1.3*self.scale * i + 8.5, 2.0), wp.quat_identity()))

            s = builder.add_shape_sphere(
                pos=(0.3, 0.0, 0.0),
                radius=0.25*self.scale, 
                density=(0.0 if i == 0 else 1000.0),
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=1.0,
                restitution=self.restitution)

        # capsules
        for i in range(1, self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((0.1*i + 0.0, 12.5 + 0.4*i, 4.0), wp.quat_identity()))

            s = builder.add_shape_capsule( 
                pos=(0.0, 0.0, 0.0),
                radius=0.25*self.scale,
                half_height=self.scale*0.5,
                up_axis=0,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.5,
                density=(0.0 if i == 0 else 1000.0),
                restitution=self.restitution)

        if True:
            b = builder.add_body(origin=wp.transform((0.3, 9.2, 3.0), wp.quat_rpy(0.3, 1.2, 0.6)))
            s = builder.add_shape_box( 
                pos=(0.0, 0.0, 0.0),
                hx=1.2*self.scale,
                hy=0.4*self.scale,
                hz=0.5*self.scale,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.5,
                density=1e3,
                restitution=self.restitution)

        if True:
            axis = np.array((1.0, 0.0, 0.0))
            axis /= np.linalg.norm(axis)
            builder.add_shape_plane(
                pos=(0.0, 3.5, -6.0),
                rot=wp.quat_from_axis_angle(axis, math.pi*0.15),
                width=2.5,
                length=5.0,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.5,
                restitution=self.restitution)
            builder.add_shape_plane(
                pos=(0.0, 6.5, 2.0),
                rot=wp.quat_from_axis_angle(axis, -math.pi*0.15),
                width=2.5,
                length=5.0,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.5,
                restitution=self.restitution)


        if True:
            axis = np.array((0.2, 0.1, 0.7))
            axis = axis/np.linalg.norm(axis)
            for i in range(1, self.num_bodies):
                loaded_mesh = self.load_mesh(
                    # os.path.join(os.path.dirname(__file__), f"assets/monkey.obj"))
                    os.path.join(os.path.dirname(__file__), f"assets/bowl.obj"))
                    # os.path.join(os.path.dirname(__file__), f"assets/icosphere.obj"))
                b2 = builder.add_body(
                    origin=wp.transform((1.5, 7.0 + i*1.0, 0.0), wp.quat_from_axis_angle(axis, 0.0)))
                    # origin=wp.transform((1.5, 1.7 + i*1.0, 4.0), wp.quat_from_axis_angle(axis, 0.0)))
                builder.add_shape_mesh(
                    body=b2,
                    mesh=loaded_mesh,
                    pos=(0.0, 0.0, 0.0),
                    scale=(0.4, 0.4, 0.4),
                    ke=1e6,  
                    kd=0.0,
                    kf=0.0, # 1e1,
                    mu=0.3,
                    restitution=self.restitution,
                    density=1e3,
                )

        # capsules (stable stack)
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((0.0, 8.0 + 0.4*i, 0.0), rot90s[i%2]))

            s = builder.add_shape_capsule( 
                pos=(0.3, 0.0, 0.0),
                radius=0.25*self.scale,
                half_height=self.scale*0.5,
                up_axis=0,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=1.0,
                density=1e3,
                restitution=self.restitution)


if __name__ == "__main__":
    run_env(Demo)
