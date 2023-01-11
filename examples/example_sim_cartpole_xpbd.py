# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation 
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

class Robot:

    frame_dt = 1.0/60.0

    episode_duration = 20.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    def __init__(self, render=True, num_envs=1, device='cpu'):

        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs
        articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/cartpole.urdf"),
            articulation_builder,
            xform=wp.transform(np.array((0.0, 4.0, 0.0)), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
            floating=False, 
            density=0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False)

        for i in range(num_envs):
            # articulation_builder.joint_X_p[0] = wp.transform(np.array((i*2.0, 4.0, 0.0)), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5))

            builder.add_rigid_articulation(articulation_builder, xform=wp.transform(np.array((i*2.0, 4.0, 0.0)), wp.quat_identity()))

            # joint initial positions
            builder.joint_q[-3:] = [0.0, 0.3, 0.0]

            builder.joint_target[-3:] = [0.0, 0.0, 0.0]

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = True

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.solve_iterations = 5
        self.integrator = wp.sim.XPBDIntegrator(self.solve_iterations)

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_cartpole.usd"))


    def run(self, render=True, plot=True):

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

        if (self.model.ground):
            self.model.collide(self.state)

        profiler = {}

        # create update graph
        wp.capture_begin()

        # simulate
        for i in range(0, self.sim_substeps):
            self.state.clear_forces()
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
            self.sim_time += self.sim_dt
                
        graph = wp.capture_end()

        q_history = []
        q_history.append(self.state.body_q.numpy().copy())
        qd_history = []
        qd_history.append(self.state.body_qd.numpy().copy())
        delta_history = []
        delta_history.append(self.state.body_deltas.numpy().copy())

        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):

            from tqdm import trange
            for f in trange(self.episode_frames):
                
                wp.capture_launch(graph)
                # for i in range(0, self.sim_substeps):
                #     self.state.clear_forces()
                #     self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
                #     self.sim_time += self.sim_dt
                self.sim_time += self.frame_dt

                if (self.render):

                    with wp.ScopedTimer("render", False):

                        if (self.render):
                            self.render_time += self.frame_dt
                            
                            self.renderer.begin_frame(self.render_time)
                            self.renderer.render(self.state)
                            self.renderer.end_frame()

                    self.renderer.save()

                q_history.append(self.state.body_q.numpy().copy())
                qd_history.append(self.state.body_qd.numpy().copy())
                delta_history.append(self.state.body_deltas.numpy().copy())

            wp.synchronize()

 
        avg_time = np.array(profiler["simulate"]).mean()/self.episode_frames
        avg_steps_second = 1000.0*float(self.num_envs)/avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        if plot:
            import matplotlib.pyplot as plt
            q_history = np.array(q_history)
            qd_history = np.array(qd_history)
            delta_history = np.array(delta_history)
            
            fig, ax = plt.subplots(3, 6, figsize=(10, 10))
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            ax[0,0].set_title("Cart Position")
            ax[0,0].grid()
            ax[1,0].set_title("Pole 1 Position")
            ax[1,0].grid()
            ax[2,0].set_title("Pole 2 Position")
            ax[2,0].grid()

            ax[0,1].set_title("Cart Orientation")
            ax[0,1].grid()
            ax[1,1].set_title("Pole 1 Orientation")
            ax[1,1].grid()
            ax[2,1].set_title("Pole 1 Orientation")
            ax[2,1].grid()

            ax[0,2].set_title("Cart Linear Velocity")
            ax[0,2].grid()
            ax[1,2].set_title("Pole 1 Linear Velocity")
            ax[1,2].grid()
            ax[2,2].set_title("Pole 2 Linear Velocity")
            ax[2,2].grid()

            ax[0,3].set_title("Cart Angular Velocity")
            ax[0,3].grid()
            ax[1,3].set_title("Pole 1 Angular Velocity")
            ax[1,3].grid()
            ax[2,3].set_title("Pole 2 Angular Velocity")
            ax[2,3].grid()

            ax[0,4].set_title("Cart Linear Delta")
            ax[0,4].grid()
            ax[1,4].set_title("Pole 1 Linear Delta")
            ax[1,4].grid()
            ax[2,4].set_title("Pole 2 Linear Delta")
            ax[2,4].grid()

            ax[0,5].set_title("Cart Angular Delta")
            ax[0,5].grid()
            ax[1,5].set_title("Pole 1 Angular Delta")
            ax[1,5].grid()
            ax[2,5].set_title("Pole 2 Angular Delta")
            ax[2,5].grid()

            ax[0,0].plot(q_history[:,1,:3])
            ax[0,1].plot(q_history[:,1,3:])

            ax[0,2].plot(qd_history[:,1,3:])
            ax[0,3].plot(qd_history[:,1,:3])

            ax[0,4].plot(delta_history[:,1,3:])
            ax[0,5].plot(delta_history[:,1,:3])

            if q_history.shape[1] > 2:
                ax[1,0].plot(q_history[:,2,:3])
                ax[1,1].plot(q_history[:,2,3:])
                ax[1,2].plot(qd_history[:,2,3:])
                ax[1,3].plot(qd_history[:,2,:3])
                ax[1,4].plot(delta_history[:,2,3:])
                ax[1,5].plot(delta_history[:,2,:3])
            if q_history.shape[1] > 3:
                ax[2,0].plot(q_history[:,3,:3])
                ax[2,1].plot(q_history[:,3,3:])
                ax[2,2].plot(qd_history[:,3,3:])
                ax[2,3].plot(qd_history[:,3,:3])
                ax[2,4].plot(delta_history[:,3,3:])
                ax[2,5].plot(delta_history[:,3,:3])

            plt.show()

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

    robot = Robot(render=True, device=wp.get_preferred_device(), num_envs=2)
    robot.run()
