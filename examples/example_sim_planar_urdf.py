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
wp.config.verify_fp = False
wp.config.mode = "debug"
wp.config.cache_kernels = True

# clear cache
import os, shutil
folder = r'C:\Users\eric-\AppData\Local\NVIDIA Corporation\warp\Cache\0.2.2\bin'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

class Robot:

    frame_dt = 1.0/100.0

    episode_duration = 1.5  # 3.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 20
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    def __init__(self, render=True, num_envs=1, device='cpu'):

        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs

        self.system_name = "hopper"

        for i in range(num_envs):
            wp.sim.parse_urdf(os.path.join(os.path.dirname(__file__), f"assets/{self.system_name}.urdf"), 
                builder,
                xform=wp.transform(np.array([(i//10)*1.0, 0.70, (i%10)*1.0]), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
                floating=False,
                density=1000,
                armature=0.01,
                stiffness=0.0,  # 120,
                damping=10,
                shape_ke=1.e+4,
                shape_kd=1.e+2,
                shape_kf=1.e+2,
                shape_mu=1.0,
                limit_ke=1.e+4,
                limit_kd=1.e+1)

            # joints
            # builder.joint_q[-12:] = [0.2, 0.4, -0.6,
            #                          -0.2, -0.4, 0.6,
            #                          -0.2, 0.4, -0.6,
            #                          0.2, -0.4, 0.6]

            # builder.joint_target[-12:] = [0.2, 0.4, -0.6,
            #                               -0.2, -0.4, 0.6,
            #                               -0.2, 0.4, -0.6,
            #                               0.2, -0.4, 0.6]

            # builder.joint_target_ke = np.zeros_like(builder.joint_target_ke)
        np.set_printoptions(suppress=True)

        builder.body_com = np.zeros_like(builder.body_com)

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = True

        self.model.joint_attach_ke = 16000.0
        self.model.joint_attach_kd = 200.0

        self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.solve_iterations = 15
        # self.relaxation = 0.1
        # self.integrator = wp.sim.XPBDIntegrator(self.solve_iterations, self.relaxation)

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(
                self.model,
                os.path.join(os.path.dirname(__file__), f"outputs/example_sim_{self.system_name}.usd"))


    def run(self, render=True):

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
        # wp.capture_begin()

        from tqdm import trange

        # simulate
        # for i in range(self.sim_substeps):
        #     self.state.clear_forces()
        #     self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
        #     self.sim_time += self.sim_dt
                
        # graph = wp.capture_end()

        

        
        q_history = []
        q_history.append(self.state.body_q.numpy().copy())
        qd_history = []
        qd_history.append(self.state.body_qd.numpy().copy())
        delta_history = []
        delta_history.append(self.state.body_deltas.numpy().copy())


        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):

            for f in trange(0, self.episode_frames):
                
                # wp.capture_launch(graph)

                for i in range(self.sim_substeps):
                    self.state.clear_forces()
                    self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
                    self.sim_time += self.sim_dt
                    
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

        
        import matplotlib.pyplot as plt
        q_history = np.array(q_history)
        qd_history = np.array(qd_history)
        delta_history = np.array(delta_history)

        fig, ax = plt.subplots(self.model.body_count, 6, figsize=(10, 10), squeeze=False, sharex=True)
        fig.subplots_adjust(hspace=0.2, wspace=0.2, left=0, bottom=0, right=1, top=1)
        for i in range(self.model.body_count):
            ax[i,0].set_title(f"Body {i} Position")
            ax[i,0].grid()
            ax[i,1].set_title(f"Body {i} Orientation")
            ax[i,1].grid()
            ax[i,2].set_title(f"Body {i} Linear Velocity")
            ax[i,2].grid()
            ax[i,3].set_title(f"Body {i} Angular Velocity")
            ax[i,3].grid()
            ax[i,4].set_title(f"Body {i} Linear Delta")
            ax[i,4].grid()
            ax[i,5].set_title(f"Body {i} Angular Delta")
            ax[i,5].grid()
            ax[i,0].plot(q_history[:,i,:3])        
            ax[i,1].plot(q_history[:,i,3:])
            ax[i,2].plot(qd_history[:,i,3:])
            ax[i,3].plot(qd_history[:,i,:3])
            ax[i,4].plot(delta_history[:,i,3:])
            ax[i,5].plot(delta_history[:,i,:3])
            ax[i,0].set_xlim(0, self.sim_steps)
            ax[i,1].set_xlim(0, self.sim_steps)
            ax[i,2].set_xlim(0, self.sim_steps)
            ax[i,3].set_xlim(0, self.sim_steps)
            ax[i,4].set_xlim(0, self.sim_steps)
            ax[i,5].set_xlim(0, self.sim_steps)
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

    device = wp.get_preferred_device()
    device = "cpu"

    robot = Robot(render=True, device=device, num_envs=2)
    robot.run()
