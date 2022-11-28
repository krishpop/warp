# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid FEM
#
# Shows how to set up a rigid sphere colliding with an FEM beam
# using wp.sim.ModelBuilder().
#
###########################################################################

import os

import numpy as np
import warp as wp

import warp.sim
import warp.sim.render

from tqdm import trange

wp.init()


class Example:

    def __init__(self, stage):

        self.sim_width = 8
        self.sim_height = 8

        self.sim_fps = 60.0
        self.sim_substeps = 5
        self.sim_duration = 5.0
        self.sim_frames = int(self.sim_duration*self.sim_fps)
        self.sim_dt = (1.0/self.sim_fps)/self.sim_substeps
        self.sim_time = 0.0
        self.sim_iterations = 1
        self.sim_relaxation = 1.0

        builder = wp.sim.ModelBuilder()

        builder.add_soft_grid(
            pos=(0.0, 0.1, 0.0), 
            rot=wp.quat_identity(), 
            vel=(0.0, 0.0, 0.0), 
            dim_x=20, 
            dim_y=10, 
            dim_z=10,
            cell_x=0.1, 
            cell_y=0.1,
            cell_z=0.1,
            density=100.0, 
            k_mu=50000.0, 
            k_lambda=20000.0,
            k_damp=0.0,
            fix_left=False)

        for _ in range(500):
            xyz = np.random.randn(3)
            builder.add_particle(pos=(xyz[0]*0.25, 3.5 + xyz[1], xyz[2]*0.25), vel=(0.0, 0.0, 0.0), mass=1.0)

        builder.add_body(origin=wp.transform((0.5, 2.0, 0.5), wp.quat_identity()))
        builder.add_shape_sphere(body=0, radius=0.75, density=0.1)

        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_distance = 0.05
        self.model.soft_contact_ke = 1.e+3
        self.model.soft_contact_mu = 0.2
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_kf = 1.e+1

        self.integrator = wp.sim.XPBDIntegrator(iterations=60, soft_body_relaxation=0.4)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=100.0)

    def update(self):

        with wp.ScopedTimer("simulate", active=False):
            for s in range(self.sim_substeps):

                wp.sim.collide(self.model, self.state_0)

                self.state_0.clear_forces()
                self.state_1.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.sim_time += self.sim_dt

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def render(self, is_live=False):

        with wp.ScopedTimer("render", active=False): 
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == '__main__':
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_fem_xpbd.usd")

    example = Example(stage_path)

    wp.capture_begin()
    example.update()
    graph = wp.capture_end()

    for i in trange(example.sim_frames):
        example.sim_time = i * example.sim_dt * example.sim_substeps
        # example.update()
        wp.capture_launch(graph)
        example.render()

    example.renderer.save()
