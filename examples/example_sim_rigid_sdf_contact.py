# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example NanoVDB
#
# Shows how to implement a particle simulation with collision against
# a NanoVDB signed-distance field. In this example the NanoVDB field
# is created offline in Houdini. The particle kernel uses the Warp
# wp.volume_sample_f() method to compute the SDF and normal at a point.
#
###########################################################################

import os
import math

import numpy as np
import warp as wp

wp.config.verify_fp = False
wp.config.mode = 'release'

import warp.sim
import warp.sim.render

from tqdm import trange

use_meshio = True
if use_meshio:
    import meshio
else:
    import openmesh

wp.init()


class Example:

    def load_volume(self, filename):
        if os.path.exists(filename):
            file = open(filename, "rb")
            return wp.Volume.load_from_nvdb(file, device=self.device)
        else:
            print("Could not find NVDB volume at {}, skipping.".format(filename))
            return None

    def load_mesh(self, filename):
        if use_meshio:
            m = meshio.read(filename)
            mesh_points = np.array(m.points)
            mesh_indices = np.array(m.cells[0].data, dtype=np.int32).flatten()
        else:
            m = openmesh.read_trimesh(filename)
            mesh_points = np.array(m.points())
            mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32).flatten()
        return wp.sim.Mesh(mesh_points, mesh_indices)

    def __init__(self, stage):

        self.sim_steps = 500
        self.sim_dt = 1.0/120.0
        self.sim_substeps = 10

        self.sim_time = 0.0
        self.sim_timers = {}
        self.sim_render = True

        self.sim_restitution = 0.0
        self.sim_margin = 15.0

        self.device = wp.get_preferred_device()

        # model_name = "icosphere"
        model_name = "monkey"


        bowl_volume = self.load_volume(os.path.join(os.path.dirname(__file__), f"assets/bowl.nvdb"))
        bowl_mesh = self.load_mesh(os.path.join(os.path.dirname(__file__), f"assets/bowl.obj"))

        body1_volume = self.load_volume(os.path.join(os.path.dirname(__file__), f"assets/{model_name}.nvdb"))
        body1_mesh = self.load_mesh(os.path.join(os.path.dirname(__file__), f"assets/{model_name}.obj"))

        body2_volume = self.load_volume(os.path.join(os.path.dirname(__file__), f"assets/{model_name}.nvdb"))
        body2_mesh = self.load_mesh(os.path.join(os.path.dirname(__file__), f"assets/{model_name}.obj"))

        body3_volume = self.load_volume(os.path.join(os.path.dirname(__file__), f"assets/{model_name}.nvdb"))
        body3_mesh = self.load_mesh(os.path.join(os.path.dirname(__file__), f"assets/{model_name}.obj"))

        builder = wp.sim.ModelBuilder()

        restitution = 1.0

        bowl = builder.add_body(
            origin=wp.transform((0.0, 1.1, 0.0), wp.quat_from_axis_angle((1.0, 0.0, 0.0), 0.01))
        )
        builder.add_shape_mesh(
            body=bowl,
            mesh=bowl_mesh,
            volume=bowl_volume,
            pos=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            ke=1e6,
            kd=0.0, # 1e2,
            kf=0.0, # 1e1,
            mu=0.3,
            restitution=restitution,
            density=1e3,
        )

        b1 = builder.add_body(
            origin=wp.transform((-0.8, 1.7, -0.8), wp.quat_identity())
        )
        builder.add_shape_mesh(
            body=b1,
            mesh=body1_mesh,
            volume=body1_volume,
            pos=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            ke=1e6,
            kd=0.0,
            kf=0.0, # 1e1, 
            mu=0.3,
            restitution=restitution,
            density=1e3,
        )

        axis = np.array((0.2, 0.1, 0.7))
        axis = axis/np.linalg.norm(axis)
        b2 = builder.add_body(
            origin=wp.transform((0.5, 2.1, 0.6), wp.quat_from_axis_angle(axis, -math.pi/2.0)))
        builder.add_shape_mesh(
            body=b2,
            mesh=body2_mesh,
            volume=body2_volume,
            pos=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            ke=1e6,  
            kd=0.0,
            kf=0.0, # 1e1,
            mu=0.3,
            restitution=restitution,
            density=1e3,
        )

        # axis = np.array((0.2, 0.8, 0.17))
        # axis = axis/np.linalg.norm(axis)
        # b3 = builder.add_body(
        #     origin=wp.transform((0.0, 4.2, -0.3), wp.quat_from_axis_angle(axis, -math.pi/2.0)))
        # builder.add_shape_mesh(
        #     body=b3,
        #     mesh=body3_mesh,
        #     volume=body3_volume,
        #     pos=(0.0, 0.0, 0.0),
        #     scale=(1.0, 1.0, 1.0),
        #     ke=1e6,
        #     kd=1e2,
        #     kf=0.0, # 1e1,
        #     density=1e3,
        # )

        self.model = builder.finalize(self.device)
        self.model.ground = True

        # make sure we allocate enough rigid contacts
        self.model.allocate_rigid_contacts(2**14)

        self.integrator = wp.sim.XPBDIntegrator(
            iterations=1,
            contact_con_weighting=True,
            contact_normal_relaxation=1.0)
        # self.integrator = wp.sim.SemiImplicitIntegrator()
        self.state = self.model.state()

        # one time collide for ground contact
        with wp.ScopedTimer("collide", active=True):
            self.model.collide(self.state)

        self.renderer = wp.sim.render.SimRenderer(
            self.model, stage, scaling=100.0)

        self.points_a = []
        self.points_b = []
        self.max_contact_count = 1000

        self.use_cuda_graphs = True  # speed up simulation via CUDA graphs

    def update(self):

        with wp.ScopedTimer("simulate", active=False):
            
            for i in range(self.sim_substeps):
                self.state.clear_forces()
                wp.sim.collide(self.model, self.state, experimental_sdf_collision=True)

                
                if not self.use_cuda_graphs and i == 0:                    
                    qs = self.state.body_q.numpy()
                    rigid_contact_count = self.model.rigid_contact_count.numpy()[0]
                    self.max_contact_count = max(self.max_contact_count, rigid_contact_count)

                    self.points_a = np.zeros((self.max_contact_count, 3))
                    self.points_b = np.zeros((self.max_contact_count, 3))

                    body_a = self.model.rigid_contact_body0.numpy()[:rigid_contact_count]
                    body_b = self.model.rigid_contact_body1.numpy()[:rigid_contact_count]

                    if rigid_contact_count > 0:
                        contact_points_a = self.model.rigid_contact_point0.numpy()
                        self.points_a[:rigid_contact_count] = [wp.transform_point(qs[body], wp.vec3(*contact_points_a[i])) for i, body in enumerate(body_a)]

                        contact_points_b = self.model.rigid_contact_point1.numpy()
                        self.points_b[:rigid_contact_count] = [wp.transform_point(qs[body], wp.vec3(*contact_points_b[i])) for i, body in enumerate(body_b)]

                    self.render()


                self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt/self.sim_substeps)   


    def render(self, is_live=False):

        with wp.ScopedTimer("render", active=False):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state)

            self.renderer.render_points("contact_points_a", np.array(self.points_a), radius=0.05)
            self.renderer.render_points("contact_points_b", np.array(self.points_b), radius=0.05)

            self.renderer.end_frame()
        
        self.sim_time += self.sim_dt

if __name__ == '__main__':
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_rigid_sdf_contact.usd")

    example = Example(stage_path)
    # example.render()
    
    q_history = []
    q_history.append(example.state.body_q.numpy().copy())
    qd_history = []
    qd_history.append(example.state.body_qd.numpy().copy())
    delta_history = []
    delta_history.append(example.state.body_deltas.numpy().copy())
    num_con_history = []
    num_con_history.append(example.model.rigid_contact_inv_weight.numpy().copy())

    if example.use_cuda_graphs:
        wp.capture_begin()
        example.update()
        graph = wp.capture_end()

    for i in trange(example.sim_steps):
        if example.use_cuda_graphs:
            wp.capture_launch(graph)
        else:
            example.update()
        example.render()
        
        q_history.append(example.state.body_q.numpy().copy())
        qd_history.append(example.state.body_qd.numpy().copy())
        delta_history.append(example.state.body_deltas.numpy().copy())
        num_con_history.append(example.model.rigid_contact_inv_weight.numpy().copy())

    example.renderer.save()

    if True:
        import matplotlib.pyplot as plt
        q_history = np.array(q_history)
        qd_history = np.array(qd_history)
        delta_history = np.array(delta_history)
        num_con_history = np.array(num_con_history)

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
            ax[i,6].set_title(f"Body {i} Contacts")
            ax[i,6].grid()
            ax[i,0].plot(q_history[:,i,:3])        
            ax[i,1].plot(q_history[:,i,3:])
            ax[i,2].plot(qd_history[:,i,3:])
            ax[i,3].plot(qd_history[:,i,:3])
            ax[i,4].plot(delta_history[:,i,3:])
            ax[i,5].plot(delta_history[:,i,:3])
            ax[i,4].plot(delta_history[:,i,3:])
            ax[i,5].plot(delta_history[:,i,:3])
            ax[i,6].plot(num_con_history[:,i], label="Contact constraints")
            ax[i,0].set_xlim(0, example.sim_steps)
            ax[i,1].set_xlim(0, example.sim_steps)
            ax[i,2].set_xlim(0, example.sim_steps)
            ax[i,3].set_xlim(0, example.sim_steps)
            ax[i,4].set_xlim(0, example.sim_steps)
            ax[i,5].set_xlim(0, example.sim_steps)
            ax[i,6].set_xlim(0, example.sim_steps)
        plt.show()