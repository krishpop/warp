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
import torch


import warp as wp
# wp.config.mode = "debug"
# wp.config.verify_cuda = True
# wp.config.verify_fp = True
import warp.sim
import warp.sim.render
import warp.sim.tiny_render

wp.init()

class Example:

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

    def __init__(self, stage):

        self.sim_steps = 1000
        self.sim_dt = 1.0/30.0
        self.sim_time = 0.0
        self.sim_substeps = 5

        self.solve_iterations = 1
        self.relaxation = 1.0

        self.num_bodies = 5
        self.scale = 0.5
        self.ke = 1.e+5
        self.kd = 250.0
        self.kf = 500.0
        self.mu  = 1.0

        self.device = wp.get_preferred_device()

        self.plot = True

        builder = wp.sim.ModelBuilder()

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
        
        self.model = builder.finalize(self.device)
        self.model.ground = True
        self.model.rigid_contact_rolling_friction = 0.05
        self.model.rigid_contact_torsion_friction = 0.05
        self.model.rigid_contact_margin = 0.05

        self.integrator = wp.sim.XPBDIntegrator(self.solve_iterations, enable_restitution=True)
        # self.integrator = wp.sim.SemiImplicitIntegrator()
        self.integrator.contact_con_weighting = False

        self.state = self.model.state()

        # self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=10.0)
        self.renderer = wp.sim.tiny_render.TinyRenderer(self.model, stage, scaling=0.4, start_paused=True)

    def update(self):

        with wp.ScopedTimer("simulate", active=False):
            
            for i in range(self.sim_substeps):
                self.state.clear_forces()
                wp.sim.collide(self.model, self.state)
                self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt/self.sim_substeps)   

    def render(self, is_live=False):

        with wp.ScopedTimer("render", active=False):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state)
            self.renderer.end_frame()
        
        self.sim_time += self.sim_dt


if __name__ == '__main__':
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_funnel_granular.usd")

    example = Example(stage_path)

    
    from tqdm import trange
    wp.capture_begin()
    example.update()
    graph = wp.capture_end()
    
    example.render()
    for i in trange(example.sim_steps):
        wp.capture_launch(graph)
        # example.update()
        example.render()

      
    example.renderer.save()


