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
wp.config.mode == "debug"
import warp.sim
import warp.sim.render

# wp.config.verify_fp = True
wp.init()
# wp.config.mode == "debug"


class Example:

    def __init__(self, stage):

        self.sim_steps = 500
        self.sim_dt = 1.0/30.0
        self.sim_time = 0.0
        self.sim_substeps = 5

        self.solve_iterations = 1
        self.relaxation = 1.0

        self.num_bodies = 1
        self.scale = 0.5
        self.ke = 1.e+5
        self.kd = 250.0
        self.kf = 500.0

        # self.device = wp.get_preferred_device()
        self.device = "cpu"

        self.plot = True

        builder = wp.sim.ModelBuilder()

        restitution = 1.0

        # boxes
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((i, 0.0, 0.0), wp.quat_identity()))
            # b = builder.add_body(origin=wp.transform((i, 0.6, 0.0), wp.quat_rpy(0.2, 0.5, 0.8)))

            s = builder.add_shape_box( 
                pos=(0.0, 0.0, 0.0),
                hx=0.25*self.scale,
                hy=0.12*self.scale,
                hz=0.12*self.scale,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.05,
                restitution=restitution)

        # spheres
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((0.01 * i, 1.3*self.scale * i + 0.5, 1.0), wp.quat_identity()))

            s = builder.add_shape_sphere(
                pos=(0.0, 0.0, 0.0),
                radius=0.25*self.scale,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=1.0,
                restitution=restitution)

        # capsules
        for i in range(self.num_bodies):
            
            # b = builder.add_body(origin=wp.transform((0.0, 2.0, 4.0), wp.quat_rpy(0.1, 0.0, 0.3)))
            b = builder.add_body(origin=wp.transform((0.0, 2.0, 2.0), wp.quat_identity()))

            s = builder.add_shape_capsule( 
                pos=(0.0, 0.0, 0.0),
                radius=0.25*self.scale,
                half_width=self.scale*0.5,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=0.01,
                restitution=restitution)

        # initial spin 
        # for i in range(len(builder.body_qd)):
        #     # builder.body_qd[i] = (0.0, 2.0, 10.0, -1.5, 0.0, 0.0)
        #     builder.body_qd[i] = (10.0, 0.0, 0.0, 0.3, 2.5, -4.0)

        # ground_angle = np.deg2rad(30.0)
        # builder.ground = [0.0, np.sin(ground_angle), np.cos(ground_angle), -0.5]
        builder.ground = [0.0, 1.0, 0.0, 0.0]
        
        self.model = builder.finalize(self.device)
        self.model.ground = True

        self.integrator = wp.sim.XPBDIntegrator(self.solve_iterations)
        self.integrator.contact_con_weighting = False
        # self.integrator = wp.sim.SemiImplicitIntegrator()
        self.state = self.model.state()

        # one time collide for ground contact
        self.model.collide(self.state)

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=100.0)

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
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact_xpbd.usd")

    example = Example(stage_path)

    
    q_history = []
    q_history.append(example.state.body_q.numpy().copy())
    qd_history = []
    qd_history.append(example.state.body_qd.numpy().copy())
    delta_history = []
    delta_history.append(example.state.body_deltas.numpy().copy())
    num_con_history = []
    num_con_history.append(example.model.rigid_contact_inv_weight.numpy().copy())

    from tqdm import trange
    example.render()
    for i in trange(example.sim_steps):
        example.update()
        example.render()

        q_history.append(example.state.body_q.numpy().copy())
        qd_history.append(example.state.body_qd.numpy().copy())
        delta_history.append(example.state.body_deltas.numpy().copy())
        num_con_history.append(example.model.rigid_contact_inv_weight.numpy().copy())

    example.renderer.save()

    if example.plot:
        import matplotlib.pyplot as plt
        q_history = np.array(q_history)
        qd_history = np.array(qd_history)
        delta_history = np.array(delta_history)
        num_con_history = np.array(num_con_history)
        # print("max num_con_history:", np.max(num_con_history))

        fig, ax = plt.subplots(example.model.body_count, 7, figsize=(10, 10), squeeze=False)
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        for i in range(example.model.body_count):
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
            ax[i,6].set_title(f"Body {i} Num Contacts")
            ax[i,6].grid()
            ax[i,0].plot(q_history[:,i,:3])        
            ax[i,1].plot(q_history[:,i,3:])
            ax[i,2].plot(qd_history[:,i,3:])
            ax[i,3].plot(qd_history[:,i,:3])
            ax[i,4].plot(delta_history[:,i,3:])
            ax[i,5].plot(delta_history[:,i,:3])
            ax[i,6].plot(num_con_history[:,i])
            ax[i,0].set_xlim(0, example.sim_steps)
            ax[i,1].set_xlim(0, example.sim_steps)
            ax[i,2].set_xlim(0, example.sim_steps)
            ax[i,3].set_xlim(0, example.sim_steps)
            ax[i,4].set_xlim(0, example.sim_steps)
            ax[i,5].set_xlim(0, example.sim_steps)
            ax[i,6].set_xlim(0, example.sim_steps)
            ax[i,6].yaxis.get_major_locator().set_params(integer=True)
        plt.show()



