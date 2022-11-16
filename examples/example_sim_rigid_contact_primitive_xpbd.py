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
# wp.config.mode == "release"
# wp.config.verify_fp = False
# wp.config.mode == "debug"
# wp.config.cache_kernels = True
# wp.config.print_launches = False
import warp.sim
import warp.sim.render

# wp.config.verify_fp = True
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
        self.sim_dt = 1.0/60.0
        self.sim_time = 0.0
        self.sim_substeps = 8

        self.solve_iterations = 2

        self.num_bodies = 20
        self.scale = 0.5
        self.ke = 1.e+5 
        self.kd = 250.0
        self.kf = 500.0
        self.restitution = 0.5

        self.device = wp.get_preferred_device()
        # self.device = "cpu"

        self.plot = False

        builder = wp.sim.ModelBuilder()

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
                half_width=self.scale*0.5,
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
                half_width=self.scale*0.5,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=1.0,
                density=1e3,
                restitution=self.restitution)

        builder.ground = True
        
        self.model = builder.finalize(self.device)

        self.integrator = wp.sim.XPBDIntegrator(
            self.solve_iterations,
            contact_normal_relaxation=1.0)
        self.integrator.contact_con_weighting = True
        # self.integrator = wp.sim.SemiImplicitIntegrator()
        self.state = self.model.state()

        self.integrator.enable_restitution = True
        # distance threshold at which contacts are generated
        self.model.rigid_contact_margin = 0.05

        self.viz_contact_count = 0
        self.points_a = np.zeros((self.viz_contact_count, 3))
        self.points_b = np.zeros((self.viz_contact_count, 3))

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=100.0)

    def update(self):

        with wp.ScopedTimer("simulate", active=False):
            
            for i in range(self.sim_substeps):
                self.state.clear_forces()
                wp.sim.collide(self.model, self.state)

                if i == 0:
                    if self.viz_contact_count > 0:
                        # visualize contact points
                        rigid_contact_count = min(self.model.rigid_contact_count.numpy()[0], self.viz_contact_count)
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
                        
                    self.render()

                self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt/self.sim_substeps)   

    def render(self, is_live=False):

        with wp.ScopedTimer("render", active=False):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state)

            if self.viz_contact_count > 0:
                self.renderer.render_points("contact_points_a", self.points_a, radius=0.025)
                self.renderer.render_points("contact_points_b", self.points_b, radius=0.025)

                normals = self.model.rigid_contact_normal.numpy()
                for i in range(len(self.points_a)):
                    p = self.points_a[i]
                    if np.any(p != 0.0):
                        self.renderer.render_line_strip(f"normal_{i}", [p, p + 0.3 * normals[i]], color=(1.0, 0.0, 0.0), radius=0.005)
                    else:
                        # disable
                        self.renderer.render_line_strip(f"normal_{i}", [p, p], color=(1.0, 0.0, 0.0), radius=0.005)

            self.renderer.end_frame()
        
        self.sim_time += self.sim_dt


if __name__ == '__main__':
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact_primitive_xpbd.usd")

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
    for i in trange(example.sim_steps):
        example.update()

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



