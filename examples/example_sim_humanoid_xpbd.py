# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Humanoid
#
# Shows how to set up a simulation of a rigid-body Humanoid articulation based
# on the OpenAI gym environment using the wp.sim.ModelBuilder() and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################


import os
import math

import numpy as np

import warp as wp
wp.config.mode = 'debug'
import warp.sim
import warp.sim.render
from warp.sim.integrator_xpbd import update_body_contact_weights

wp.init()

class Robot:

    frame_dt = 1.0/ (60.0)

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

        self.render = render

        self.num_envs = num_envs

        # number of contact points to visualize
        self.max_contact_count = 256

        self.points_a = np.zeros((self.max_contact_count, 3), dtype=np.float32)
        self.points_b = np.zeros((self.max_contact_count, 3), dtype=np.float32)

        wp.sim.parse_mjcf(
            # r"C:\Users\eric-\Downloads\claw_warp.xml",
            os.path.join(os.path.dirname(__file__), "assets/nv_humanoid.xml"),
            # os.path.join(os.path.dirname(__file__), "assets/nv_ujoint.xml"),
            articulation_builder,
            stiffness=0.0,
            damping=0.1,
            armature=0.007,
            armature_scale=10.0,
            contact_ke=1.e+4,
            contact_kd=1.e+2,
            contact_kf=1.e+2,
            contact_mu=0.5,
            contact_restitution=0.0,
            limit_ke=1.e+2,
            limit_kd=1.e+1,
            enable_self_collisions=True)

        for i in range(num_envs):
            builder.add_rigid_articulation(articulation_builder)
            if i == 0:
                self.dof_q = len(builder.joint_q)
                self.dof_qd = len(builder.joint_qd)
 
            coord_count = 28 
            dof_count = 27
            
            coord_start = i*coord_count
            dof_start = i*dof_count

            # position above ground and rotate to +y up
            builder.joint_q[coord_start:coord_start+3] = [i*2.0 + 2.3, 1.70, 1.2]
            builder.joint_q[coord_start+3:coord_start+7] = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)

            # throw sideways
            builder.joint_qd[coord_start+3:coord_start+6] = [0.0, 0.0, 1.0]

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = True
        self.model.joint_attach_ke *= 8.0
        self.model.joint_attach_kd *= 2.0

        self.model.rigid_contact_margin = 0.1
        self.model.allocate_rigid_contacts(self.max_contact_count)

        print("joint_target: ", self.model.joint_target.numpy())
        print("joint_target_ke: ", self.model.joint_target_ke.numpy())

        # self.integrator = wp.sim.SemiImplicitIntegrator()
        self.integrator = wp.sim.XPBDIntegrator(
            iterations=1,
            joint_positional_relaxation=0.7,
            joint_angular_relaxation=0.4,
            contact_normal_relaxation=1.0)
        # self.integrator.contact_con_weighting = False
        # self.integrator = wp.sim.XPBDIntegrator()

        print("Collision filters:")
        print(builder.shape_collision_filter_pairs)
        print("Collision mask:")
        print(self.model.shape_collision_mask.numpy())

        self.requires_grad = False

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_humanoid.usd"), scaling=100.0)

 
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

        # # simulate
        # for i in range(0, self.sim_substeps):
        #     self.state.clear_forces()
        #     wp.sim.collide(self.model, self.state)
        #     self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
        #     self.sim_time += self.sim_dt
                
        # graph = wp.capture_end()
    
        q_history = []
        q_history.append(self.state.body_q.numpy().copy())
        qd_history = []
        qd_history.append(self.state.body_qd.numpy().copy())
        delta_history = []
        delta_history.append(self.state.body_deltas.numpy().copy())
        num_con_history = []
        num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())

        joint_q_history = []
        joint_q = wp.zeros_like(self.model.joint_q)
        joint_qd = wp.zeros_like(self.model.joint_qd)


        # simulate 
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):

            if (self.render):
 
                with wp.ScopedTimer("render", False):

                    if (self.render):
                        self.render_time += self.frame_dt
                        
                        self.renderer.begin_frame(self.render_time)
                        self.renderer.render(self.state)
                        self.renderer.end_frame()

                self.renderer.save()

            from tqdm import trange
            for f in trange(0, self.episode_frames):
                
                for i in range(0, self.sim_substeps):
                    self.state.clear_forces()
                    
                    # keep floating base fixed
                    # wp.sim.eval_ik(self.model, self.state, joint_q, joint_qd)
                    # q = joint_q.numpy().copy()
                    # q[:3] = [2.3, 1.70, 1.2]
                    # q[3:7] = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)
                    # qd = joint_qd.numpy().copy()
                    # qd[:6] = 0.0
                    # wp.sim.eval_fk(
                    #     self.model,
                    #     wp.array(q, dtype=wp.float32, device=joint_q.device),
                    #     wp.array(qd, dtype=wp.float32, device=joint_q.device),
                    #     None,
                    #     self.state)
                    
                    random_actions = False
                    if (random_actions):
                        act = np.zeros(len(self.model.joint_qd))
                        scale = np.array([200.0,
                                        200.0,
                                        200.0,
                                        200.0,
                                        200.0,
                                        600.0,
                                        400.0,
                                        100.0,
                                        100.0,
                                        200.0,
                                        200.0,
                                        600.0,
                                        400.0,
                                        100.0,
                                        100.0,
                                        100.0,
                                        100.0,
                                        200.0,
                                        100.0,
                                        100.0,
                                        200.0])
                        for j in range(self.num_envs):
                            act[j*self.dof_qd+6:(j+1)*self.dof_qd] = np.clip((np.random.rand(self.dof_qd-6)*2.0 - 1.0)*1000.0, a_min=-1.0, a_max=1.0)*scale*0.35

                        # act[6:] = np.clip((np.random.rand(len(self.model.joint_qd)-6)*2.0 - 1.0)*1000.0, a_min=-1.0, a_max=1.0)*scale*0.35
                        self.model.joint_act.assign(act)

                    wp.sim.collide(self.model, self.state)

                    if self.requires_grad:
                        rigid_contact_inv_weight = wp.zeros_like(self.model.rigid_contact_inv_weight)
                        rigid_active_contact_distance = wp.zeros_like(self.model.rigid_active_contact_distance)
                        rigid_active_contact_point0 = wp.empty_like(self.model.rigid_active_contact_point0, requires_grad=True)
                        rigid_active_contact_point1 = wp.empty_like(self.model.rigid_active_contact_point1, requires_grad=True)
                    else:
                        rigid_contact_inv_weight = self.model.rigid_contact_inv_weight
                        rigid_active_contact_distance = self.model.rigid_active_contact_distance
                        rigid_active_contact_point0 = self.model.rigid_active_contact_point0
                        rigid_active_contact_point1 = self.model.rigid_active_contact_point1
                        rigid_contact_inv_weight.zero_()
                        rigid_active_contact_distance.zero_()

                    wp.launch(kernel=update_body_contact_weights,
                        dim=self.model.rigid_contact_max,
                        inputs=[
                            self.state.body_q,
                            0,
                            self.model.rigid_contact_count,
                            self.model.rigid_contact_body0,
                            self.model.rigid_contact_body1,
                            self.model.rigid_contact_point0,
                            self.model.rigid_contact_point1,
                            self.model.rigid_contact_normal,
                            self.model.rigid_contact_thickness,
                            self.model.rigid_contact_shape0,
                            self.model.rigid_contact_shape1,
                            self.model.shape_transform
                        ],
                        outputs=[
                            rigid_contact_inv_weight,
                            rigid_active_contact_point0,
                            rigid_active_contact_point1,
                            rigid_active_contact_distance,
                        ],
                        device=self.model.device)

                    if (i == 0):
                        # remember the contacts from the first iteration                            
                        if self.requires_grad:
                            self.model.rigid_active_contact_distance_prev = wp.clone(rigid_active_contact_distance)
                            self.model.rigid_active_contact_point0_prev = wp.clone(rigid_active_contact_point0)
                            self.model.rigid_active_contact_point1_prev = wp.clone(rigid_active_contact_point1)
                            self.model.rigid_contact_inv_weight_prev = wp.clone(rigid_contact_inv_weight)
                        else:
                            self.model.rigid_active_contact_distance_prev.assign(rigid_active_contact_distance)
                            self.model.rigid_active_contact_point0_prev.assign(rigid_active_contact_point0)
                            self.model.rigid_active_contact_point1_prev.assign(rigid_active_contact_point1)
                            self.model.rigid_contact_inv_weight_prev.assign(rigid_contact_inv_weight)
                    

                    self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt, requires_grad=self.requires_grad)
                    self.sim_time += self.sim_dt

                    rigid_contact_count = min(self.model.rigid_contact_count.numpy()[0], self.max_contact_count)
                    self.points_a.fill(0.0)
                    self.points_b.fill(0.0)
                    self.points_a[:rigid_contact_count] = self.model.rigid_active_contact_point0_prev.numpy()[:rigid_contact_count]
                    self.points_b[:rigid_contact_count] = self.model.rigid_active_contact_point1_prev.numpy()[:rigid_contact_count]
                    
                    q_history.append(self.state.body_q.numpy().copy())
                    qd_history.append(self.state.body_qd.numpy().copy())
                    delta_history.append(self.state.body_deltas.numpy().copy())
                    num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())

                    wp.sim.eval_ik(self.model, self.state, joint_q, joint_qd)
                    joint_q_history.append(joint_q.numpy().copy())

                if (self.render):
 
                    with wp.ScopedTimer("render", False):

                        self.render_time += self.frame_dt #* 300.0
                        
                        self.renderer.begin_frame(self.render_time)
                        self.renderer.render(self.state)

                        if self.max_contact_count > 0:
                            self.renderer.render_points("contact_points_a", np.array(self.points_a), radius=0.05)
                            self.renderer.render_points("contact_points_b", np.array(self.points_b), radius=0.05)

                        self.renderer.end_frame()


            wp.synchronize()

 
        avg_time = np.array(profiler["simulate"]).mean()/self.episode_frames
        avg_steps_second = 1000.0*float(self.num_envs)/avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        if (self.render):
            self.renderer.save()

        
        if False:
            import matplotlib.pyplot as plt
            q_history = np.array(q_history)
            qd_history = np.array(qd_history)
            delta_history = np.array(delta_history)
            num_con_history = np.array(num_con_history)
            # print("max num_con_history:", np.max(num_con_history))

            body_indices = [9]

            fig, ax = plt.subplots(len(body_indices), 7, figsize=(10, 10), squeeze=False)
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            for i, j in enumerate(body_indices):
                ax[i,0].set_title(f"Body {j} Position")
                ax[i,0].grid()
                ax[i,1].set_title(f"Body {j} Orientation")
                ax[i,1].grid()
                ax[i,2].set_title(f"Body {j} Linear Velocity")
                ax[i,2].grid()
                ax[i,3].set_title(f"Body {j} Angular Velocity")
                ax[i,3].grid()
                ax[i,4].set_title(f"Body {j} Linear Delta")
                ax[i,4].grid()
                ax[i,5].set_title(f"Body {j} Angular Delta")
                ax[i,5].grid()
                ax[i,6].set_title(f"Body {j} Num Contacts")
                ax[i,6].grid()
                ax[i,0].plot(q_history[:,j,:3])        
                ax[i,1].plot(q_history[:,j,3:])
                ax[i,2].plot(qd_history[:,j,3:])
                ax[i,3].plot(qd_history[:,j,:3])
                ax[i,4].plot(delta_history[:,j,3:])
                ax[i,5].plot(delta_history[:,j,:3])
                ax[i,6].plot(num_con_history[:,j])
                ax[i,0].set_xlim(0, self.sim_steps)
                ax[i,1].set_xlim(0, self.sim_steps)
                ax[i,2].set_xlim(0, self.sim_steps)
                ax[i,3].set_xlim(0, self.sim_steps)
                ax[i,4].set_xlim(0, self.sim_steps)
                ax[i,5].set_xlim(0, self.sim_steps)
                ax[i,6].set_xlim(0, self.sim_steps)
                ax[i,6].yaxis.get_major_locator().set_params(integer=True)
            plt.show()

        if False:
            import matplotlib.pyplot as plt
            joint_q_history = np.array(joint_q_history)
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

    # robot = Robot(render=True, num_envs=1, device=wp.get_preferred_device())
    robot = Robot(render=True, num_envs=1, device="cpu")
    robot.run()
