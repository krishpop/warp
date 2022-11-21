# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Allegro
#
# Shows how to set up a simulation of a rigid-body Allegro hand articulation 
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import os
import math

import numpy as np

import warp as wp

# wp.config.mode = "debug"
# wp.config.verify_cuda = True
# wp.config.verify_fp = True

import warp.sim
import warp.sim.render


from tqdm import trange

wp.init()

class Robot:

    frame_dt = 1.0/60.0

    episode_duration = 10.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    use_graph = False

    sim_substeps = 8
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
        floating_base = False
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/isaacgymenvs/kuka_allegro_description/allegro.urdf"),
            articulation_builder,
            xform=wp.transform(np.array((0.0, 0.3, 0.0)), wp.quat_rpy(-np.pi/2, np.pi*0.75, np.pi/2)),
            floating=floating_base,
            density=1e3,
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False)
        
        # ensure all joint positions are within limits
        q_offset = (7 if floating_base else 0)
        qd_offset = (6 if floating_base else 0)
        for i in range(16):
            articulation_builder.joint_q[i+q_offset] = 0.5 * (articulation_builder.joint_limit_lower[i+qd_offset] + articulation_builder.joint_limit_upper[i+qd_offset])
            articulation_builder.joint_target[i+q_offset] = articulation_builder.joint_q[i+q_offset]
            articulation_builder.joint_target_ke[i+q_offset] = 500000.0
            articulation_builder.joint_target_kd[i+q_offset] = 500.0
        
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf"),
            articulation_builder,
            xform=wp.transform(np.array((-0.1, 0.5, 0.0)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            parse_visuals_as_colliders=False)

        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf"),
            articulation_builder,
            xform=wp.transform(np.array((0.0, 0.05, 0.05)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            parse_visuals_as_colliders=False)

        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf"),
            articulation_builder,
            xform=wp.transform(np.array((0.01, 0.15, 0.03)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            parse_visuals_as_colliders=False)
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf"),
            articulation_builder,
            xform=wp.transform(np.array((0.01, 0.05, 0.13)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            parse_visuals_as_colliders=False)

        self.bodies_per_env = len(articulation_builder.body_q)

        # box_id = len(articulation_builder.shape_geo_type)-1
        # articulation_builder.shape_collision_filter_pairs.add((0, box_id))
        # for i in range(2, 18):
        #     articulation_builder.shape_collision_filter_pairs.add((i, box_id))

        square_side = max(1, int(np.sqrt(num_envs)))
        for i in range(num_envs):
            builder.add_rigid_articulation(
                articulation_builder,
                xform=wp.transform(((i%square_side)*0.4, 0.0, (i//square_side)*0.4), wp.quat_identity()))


        # finalize model
        self.model = builder.finalize(device)
        # self.model.allocate_rigid_contacts(2**18)
        # self.model.allocate_rigid_contacts(7000)
        self.model.ground = True
        # distance threshold at which contacts are generated
        self.model.rigid_contact_margin = 0.02

        self.max_contact_count = 65
        self.points_a = np.zeros((self.max_contact_count, 3))
        self.points_b = np.zeros((self.max_contact_count, 3))

        # print("collision filters:")
        # print(builder.shape_collision_filter_pairs)

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.solve_iterations = 20
        self.integrator = wp.sim.XPBDIntegrator(self.solve_iterations)
        self.integrator.contact_con_weighting = True
        self.integrator.enable_restitution = False
        # self.integrator = wp.sim.SemiImplicitIntegrator()


        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_allegro.usd"), scaling=1000.0)


    def run(self, plot=True):

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

        # apply some motion to the hand
        body_qd = self.state.body_qd.numpy()
        for i in range(self.num_envs):
            body_qd[i*self.bodies_per_env][2] = 0.4
            body_qd[i*self.bodies_per_env][1] = 0.2
        self.state.body_qd = wp.array(body_qd, dtype=wp.spatial_vector, device=self.device)

        if (self.model.ground):
            self.model.collide(self.state)

        profiler = {}

        if (self.render):
            with wp.ScopedTimer("render", False):
                self.render_time += self.frame_dt                
                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.state)
                self.renderer.end_frame()

        if self.use_graph:
            # create update graph
            wp.capture_begin()

            # simulate
            for i in range(0, self.sim_substeps):
                self.state.clear_forces()
                wp.sim.collide(self.model, self.state)
                self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
                self.sim_time += self.sim_dt
                    
            graph = wp.capture_end()

        if plot:
            q_history = []
            q_history.append(self.state.body_q.numpy().copy())
            qd_history = []
            qd_history.append(self.state.body_qd.numpy().copy())
            delta_history = []
            delta_history.append(self.state.body_deltas.numpy().copy())

        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):

            for f in trange(self.episode_frames):
                if self.use_graph:
                    wp.capture_launch(graph)
                else:
                    for i in range(0, self.sim_substeps):
                        self.state.clear_forces()
                        wp.sim.collide(self.model, self.state)
                        self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
                        self.sim_time += self.sim_dt
                    # self.sim_time += self.frame_dtself.model.rigid_contact_inv_weight.zero_()
                    self.model.rigid_active_contact_distance.zero_()

                    # update contact points for rendering
                    # wp.launch(kernel=update_body_contact_weights,
                    #     dim=self.model.rigid_contact_max,
                    #     inputs=[
                    #         self.state.body_q,
                    #         1,
                    #         self.model.rigid_contact_count,
                    #         self.model.rigid_contact_body0,
                    #         self.model.rigid_contact_body1,
                    #         self.model.rigid_contact_point0,
                    #         self.model.rigid_contact_point1,
                    #         self.model.rigid_contact_normal,
                    #         self.model.rigid_contact_thickness,
                    #         self.model.rigid_contact_shape0,
                    #         self.model.rigid_contact_shape1,
                    #         self.model.shape_transform
                    #     ],
                    #     outputs=[
                    #         self.model.rigid_contact_inv_weight,
                    #         self.model.rigid_active_contact_point0,
                    #         self.model.rigid_active_contact_point1,
                    #         self.model.rigid_active_contact_distance,
                    #     ],
                    #     device=self.model.device)
                    # self.model.rigid_active_contact_point0_prev = self.model.rigid_active_contact_point0
                    # self.model.rigid_active_contact_point1_prev = self.model.rigid_active_contact_point1

                if (self.render):
                    # print("contacts:", self.model.rigid_contact_count.numpy()[0])
                    distance = self.model.rigid_active_contact_distance.numpy()
                    with wp.ScopedTimer("render", False):
                        rigid_contact_count = min(self.model.rigid_contact_count.numpy()[0], self.max_contact_count)
                        if not self.use_graph:
                            self.points_a.fill(0.0)
                            self.points_b.fill(0.0)
                            if rigid_contact_count > 0:
                                self.points_a[:rigid_contact_count] = self.model.rigid_active_contact_point0.numpy()[:rigid_contact_count]
                                self.points_b[:rigid_contact_count] = self.model.rigid_active_contact_point1.numpy()[:rigid_contact_count]
                                shape0 = self.model.rigid_contact_shape0.numpy()[:rigid_contact_count]
                                shape1 = self.model.rigid_contact_shape1.numpy()[:rigid_contact_count]
                                empty_contacts = np.where(np.all([shape0==-1, shape1==-1], axis=0))[0]
                                self.points_a[empty_contacts].fill(0.0)
                                self.points_b[empty_contacts].fill(0.0)
                                no_contact = np.where(distance[:rigid_contact_count]>=0.0)[0]
                                self.points_a[no_contact].fill(0.0)
                                self.points_b[no_contact].fill(0.0)


                        self.render_time += self.frame_dt
                        
                        self.renderer.begin_frame(self.render_time)
                        self.renderer.render(self.state)

                        if not self.use_graph:

                            self.renderer.render_points("contact_points_a", self.points_a, radius=0.0025)
                            self.renderer.render_points("contact_points_b", self.points_b, radius=0.0025)

                            normals = self.model.rigid_contact_normal.numpy()
                            for i in range(len(self.points_b)):
                                p = self.points_b[i]
                                if np.any(p != 0.0) and distance[i] < 0.0:
                                    self.renderer.render_line_strip(f"normal_{i}", [p, p + 0.05 * normals[i]], color=(1.0, 0.0, 0.0), radius=0.001)
                                else:
                                    # disable
                                    self.renderer.render_line_strip(f"normal_{i}", [p, p], color=(1.0, 0.0, 0.0), radius=0.001)

                        self.renderer.end_frame()

                if plot:
                    q_history.append(self.state.body_q.numpy().copy())
                    qd_history.append(self.state.body_qd.numpy().copy())
                    delta_history.append(self.state.body_deltas.numpy().copy())

            wp.synchronize()

        if (self.render):
            self.renderer.save()
 
        avg_time = np.array(profiler["simulate"]).mean()/self.episode_frames
        avg_steps_second = 1000.0*float(self.num_envs)/avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        if plot:
            import matplotlib.pyplot as plt
            q_history = np.array(q_history)
            qd_history = np.array(qd_history)
            delta_history = np.array(delta_history)

            num_viz_links = 3
            
            fig, ax = plt.subplots(num_viz_links, 6, figsize=(10, 10), squeeze=False)
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            for i in range(num_viz_links):
                qi = q_history.shape[1] - num_viz_links + i
                ax[i,0].set_title(f"Link {qi} Position")
                ax[i,0].grid()

                ax[i,1].set_title(f"Link {qi} Orientation")
                ax[i,1].grid()

                ax[i,2].set_title(f"Link {qi} Linear Velocity")
                ax[i,2].grid()

                ax[i,3].set_title(f"Link {qi} Angular Velocity")
                ax[i,3].grid()

                ax[i,4].set_title(f"Link {qi} Linear Delta")
                ax[i,4].grid()

                ax[i,5].set_title(f"Link {qi} Angular Delta")
                ax[i,5].grid()

                ax[i,0].plot(q_history[:,qi,:3])
                ax[i,1].plot(q_history[:,qi,3:])

                ax[i,2].plot(qd_history[:,qi,3:])
                ax[i,3].plot(qd_history[:,qi,:3])

                ax[i,4].plot(delta_history[:,qi,3:])
                ax[i,5].plot(delta_history[:,qi,:3])

            plt.show()

        return 1000.0*float(self.num_envs)/avg_time

profile = True

if profile:

    env_count = 2
    env_times = []
    env_size = []

    for i in range(12):

        robot = Robot(render=False, device='cuda', num_envs=env_count)
        robot.use_graph = True
        steps_per_second = robot.run(plot=False)

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

    robot = Robot(render=True, device=wp.get_preferred_device(), num_envs=100)
    # robot = Robot(render=True, device="cpu", num_envs=2)
    robot.use_graph = True
    robot.run(plot=False)
