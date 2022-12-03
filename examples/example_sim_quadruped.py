# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Quadruped
#
# Shows how to set up a simulation of a rigid-body quadruped articulation 
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import math
import numpy as np
import os

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

class Robot:

    frame_dt = 1.0/100.0

    episode_duration = 5.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 5
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    def __init__(self, render=True, num_envs=1, device=None):

        builder = wp.sim.ModelBuilder()
        articulation_builder = wp.sim.ModelBuilder()

        self.device = wp.get_device(device)
        self.render = render

        self.num_envs = num_envs

        wp.sim.parse_urdf(os.path.join(os.path.dirname(__file__), "assets/quadruped.urdf"), 
            articulation_builder,
            xform=wp.transform_identity(),
            floating=True,
            density=1000,
            armature=0.01,
            stiffness=120,
            damping=10,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1)

        for i in range(num_envs):
            
            builder.add_rigid_articulation(
                articulation_builder,
                xform=wp.transform(np.array([(i//10)*1.0, 0.70, (i%10)*1.0]), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
            )

            # joints
            builder.joint_q[-12:] = [0.2, 0.4, -0.6,
                                     -0.2, -0.4, 0.6,
                                     -0.2, 0.4, -0.6,
                                     0.2, -0.4, 0.6]

            builder.joint_target[-12:] = [0.2, 0.4, -0.6,
                                          -0.2, -0.4, 0.6,
                                          -0.2, 0.4, -0.6,
                                          0.2, -0.4, 0.6]
        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize(self.device)
        self.model.ground = True

        self.model.joint_attach_ke = 16000.0
        self.model.joint_attach_kd = 200.0

        self.integrator = wp.sim.XPBDIntegrator()

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(
                self.model,
                os.path.join(os.path.dirname(__file__), "outputs/example_sim_quadruped.usd"),
                scaling=100.0)


    def run(self):

        #---------------
        # run simulation

        self.sim_time = 0.0
        self.state = self.model.state()

        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

        profiler = {}

        # create update graph
        wp.capture_begin()

        # simulate
        for i in range(0, self.sim_substeps):
            self.state.clear_forces()
            wp.sim.collide(self.model, self.state)
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
            self.sim_time += self.sim_dt
                
        graph = wp.capture_end()


        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):

            for f in range(0, self.episode_frames):
                
                wp.capture_launch(graph)
                self.sim_time += self.frame_dt

                if (self.render):

                    with wp.ScopedTimer("render", False):

                        if (self.render):
                            self.render_time += self.frame_dt
                            
                            self.renderer.begin_frame(self.render_time)
                            self.renderer.render(self.state)
                            self.renderer.end_frame()

                    self.renderer.save()

            wp.synchronize()

 
        avg_time = np.array(profiler["simulate"]).mean()/self.episode_frames
        avg_steps_second = 1000.0*float(self.num_envs)/avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        return 1000.0*float(self.num_envs)/avg_time

profile = False

if profile:

    env_count = 2
    env_times = []
    env_size = []

    for i in range(15):

        robot = Robot(render=False, device='cuda', num_envs=env_count)
        steps_per_second = robot.run()

        env_size.append(env_count)
        env_times.append(steps_per_second)
        
        env_count *= 2

    # dump times
    for i in range(len(env_times)):
        print(f"envs: {env_size[i]} steps/second: {env_times[i]}")

    # plot
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(env_size, env_times)
    plt.xscale('log')
    plt.xlabel("Number of Envs")
    plt.yscale('log')
    plt.ylabel("Steps/Second")
    plt.show()

else:

    robot = Robot(render=True, num_envs=10)
    robot.run()
