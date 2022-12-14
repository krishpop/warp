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
import warp.sim
import warp.sim.render
import warp.sim.tiny_render

from tqdm import trange

wp.init()
# wp.set_device("cpu")

class Robot:


    def __init__(self, render=True, num_envs=1, device=None):

        builder = wp.sim.ModelBuilder()

        self.use_graph_capture = wp.get_device(device).is_cuda

        self.render = render

        self.num_envs = num_envs

        builder = wp.sim.ModelBuilder()

        articulation_builder = wp.sim.ModelBuilder()

        settings = wp.sim.parse_usd(
            # os.path.join(os.path.dirname(__file__), "assets/box_on_quad.usd"),
            # os.path.join(os.path.dirname(__file__), "assets/contact_pair_filtering.usd"),
            # os.path.join(os.path.dirname(__file__), "assets/rocks.usd"),
            # os.path.join(os.path.dirname(__file__), "assets/distance_joint.usd"),
            os.path.join(os.path.dirname(__file__), "assets/revolute_joint.usd"),
            # os.path.join(os.path.dirname(__file__), "assets/revolute_joint2.usd"),
            # os.path.join(os.path.dirname(__file__), "assets/spheres_with_materials.usd"),
            # os.path.join(os.path.dirname(__file__), "assets/chair_stacking.usd"),
            # os.path.join(os.path.dirname(__file__), "assets/material_density.usda"),
            # os.path.join(os.path.dirname(__file__), "assets/shapes_on_plane.usda"),
            # os.path.join(os.path.dirname(__file__), "assets/articulation.usda"),
            # os.path.join(os.path.dirname(__file__), "assets/ropes.usda"),
            articulation_builder,
        )
    
        self.frame_dt = 1.0 / settings["fps"]
        self.episode_duration = 15.0 #settings["duration"]
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.episode_frames = int(self.episode_duration/self.frame_dt)
        self.sim_steps = int(self.episode_duration / self.sim_dt)
        self.sim_time = 0.0
        self.render_time = 0.0

        for i in range(num_envs):
            builder.add_rigid_articulation(
                articulation_builder,
                # xform=wp.transform((i*2.0, 0, 0), wp.quat_identity()),
            )

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = False
        self.state = self.model.state()

        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

        
        joint_names = {
            wp.sim.JOINT_BALL.val : "ball",
            wp.sim.JOINT_REVOLUTE.val : "hinge",
            wp.sim.JOINT_PRISMATIC.val : "slide",
            wp.sim.JOINT_UNIVERSAL.val : "universal",
            wp.sim.JOINT_COMPOUND.val : "compound",
            wp.sim.JOINT_FREE.val : "free",
            wp.sim.JOINT_FIXED.val : "fixed",
            wp.sim.JOINT_DISTANCE.val : "distance",
            wp.sim.JOINT_D6.val : "D6",
        }

        print("COM", self.model.body_com.numpy())
        print("Mass", self.model.body_mass.numpy())
        print("Inertia", self.model.body_inertia.numpy())
        print("joint_X_p", self.model.joint_X_p.numpy())
        print("shape_transform", self.model.shape_transform.numpy())
        print("geo_scale", self.model.shape_geo_scale.numpy())
        print("collision filters", sorted(list(builder.shape_collision_filter_pairs)))
        print("joint types", [joint_names[joint_type] for joint_type in self.model.joint_type.numpy()])
        print("joint parent", self.model.joint_parent.numpy())

        self.integrator = wp.sim.XPBDIntegrator(
            enable_restitution=True,
            iterations=10
        )
        # self.integrator = wp.sim.SemiImplicitIntegrator()

        #-----------------------
        # set up Usd renderer
        if (self.render):
            if False:
                self.renderer = wp.sim.tiny_render.TinyRenderer(
                    self.model,
                    os.path.join(os.path.dirname(__file__), "outputs/example_sim_usd.usd"),
                    scaling=1.0,
                    fps=settings["fps"],
                    upaxis=settings["upaxis"],
                    start_paused=True)
            else:
                self.renderer = wp.sim.render.SimRenderer(
                    self.model,
                    os.path.join(os.path.dirname(__file__), "outputs/example_sim_usd.usd"),
                    scaling=100.0,
                    fps=settings["fps"],
                    upaxis=settings["upaxis"])

    def step(self):
        # simulate
        for i in range(self.sim_substeps):
            self.state.clear_forces()
            wp.sim.collide(self.model, self.state)
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
            self.sim_time += self.sim_dt

    def run(self):

        #---------------
        # run simulation

        self.sim_time = 0.0

        joint_q_history = []
        joint_q = wp.zeros_like(self.model.joint_q)
        joint_qd = wp.zeros_like(self.model.joint_qd)
        wp.sim.eval_ik(self.model, self.state, joint_q, joint_qd)
        q_history = [joint_q.numpy()]
        qd_history = [joint_qd.numpy()]

        self.renderer.begin_frame(self.render_time)
        self.renderer.render(self.state)
        self.renderer.end_frame()

        profiler = {}

        if self.use_graph_capture:
            # create update graph
            wp.capture_begin()
            self.step()    
            graph = wp.capture_end()


        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):

            for f in trange(0, self.episode_frames):
                
                if self.use_graph_capture:
                    wp.capture_launch(graph)
                else:
                    self.step()

                self.sim_time += self.frame_dt

                if (self.render):

                    with wp.ScopedTimer("render", False):

                        if (self.render):
                            self.render_time += self.frame_dt
                            
                            self.renderer.begin_frame(self.render_time)
                            self.renderer.render(self.state)
                            self.renderer.end_frame()

                wp.sim.eval_ik(self.model, self.state, joint_q, joint_qd)
                q_history.append(joint_q.numpy())
                qd_history.append(joint_qd.numpy())


            wp.synchronize()

 
        avg_time = np.array(profiler["simulate"]).mean()/self.episode_frames
        avg_steps_second = 1000.0*float(self.num_envs)/avg_time

        if (self.render):
            self.renderer.save()

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        if False:
            import matplotlib.pyplot as plt
            joint_q_history = np.array(q_history)
            dof_q = joint_q_history.shape[1]
            ncols = int(np.ceil(np.sqrt(dof_q)))
            nrows = int(np.ceil(dof_q / float(ncols)))
            fig, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                constrained_layout=True,
                figsize=(ncols * 3.5, nrows * 3.5),
                squeeze=False,
                sharex=True
            )

            joint_id = 0
            joint_names = {
                wp.sim.JOINT_BALL.val : "ball", 
                wp.sim.JOINT_REVOLUTE.val : "hinge", 
                wp.sim.JOINT_PRISMATIC.val : "slide", 
                wp.sim.JOINT_UNIVERSAL.val : "universal",
                wp.sim.JOINT_COMPOUND.val : "compound",
                wp.sim.JOINT_FREE.val : "free", 
                wp.sim.JOINT_FIXED.val : "fixed"
            }
            joint_lower = self.model.joint_limit_lower.numpy()
            joint_upper = self.model.joint_limit_upper.numpy()
            joint_type = self.model.joint_type.numpy()
            while joint_id < len(joint_type)-1 and joint_type[joint_id] == wp.sim.JOINT_FIXED.val:
                # skip fixed joints
                joint_id += 1
            q_start = self.model.joint_q_start.numpy()
            qd_start = self.model.joint_qd_start.numpy()
            qd_i = qd_start[joint_id]
            for dim in range(ncols * nrows):
                ax = axes[dim // ncols, dim % ncols]
                if dim >= dof_q:
                    ax.axis("off")
                    continue
                ax.grid()
                ax.plot(joint_q_history[:, dim])
                if joint_type[joint_id] != wp.sim.JOINT_FREE.val:
                    lower = joint_lower[qd_i]
                    if abs(lower) < 2*np.pi:
                        ax.axhline(lower, color="red")
                    upper = joint_upper[qd_i]
                    if abs(upper) < 2*np.pi:
                        ax.axhline(upper, color="red")
                joint_name = joint_names[joint_type[joint_id]]
                ax.set_title(f"$\\mathbf{{q_{{{dim}}}}}$ ({self.model.joint_name[joint_id]} / {joint_name} {joint_id})")
                if joint_id < self.model.joint_count-1 and q_start[joint_id+1] == dim+1:
                    joint_id += 1
                    qd_i = qd_start[joint_id]
                else:
                    qd_i += 1
            plt.tight_layout()
            plt.show()



        return 1000.0*float(self.num_envs)/avg_time

profile = False

if profile:

    env_count = 2
    env_times = []
    env_size = []

    for i in range(15):

        robot = Robot(render=False, num_envs=env_count)
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

    robot = Robot(render=True, num_envs=1)
    robot.run()
