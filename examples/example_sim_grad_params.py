# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Parameter Gradient Computation
#
# Shows how to compute gradients of simulation parameters from a rigid-body
# simulation involving contacts.
#
###########################################################################


from scipy.misc import derivative
import os
import math

import random
import torch
import numpy as np

from tqdm import trange

import warp as wp
# wp.config.mode = "debug"
import warp.sim
import warp.sim.render


# wp.config.verify_fp = True
wp.init()

@wp.kernel
def loss_kernel(body_q: wp.array(dtype=wp.transform), loss: wp.array(dtype=float, ndim=1)):
    tid = wp.tid()
    i = tid//3
    j = tid % 3
    x = wp.transform_get_translation(body_q[i])
    loss[i*3+j] = x[j]


@wp.kernel
def set_mass(mass: wp.array(dtype=wp.float32),
             box_size: wp.array(dtype=wp.float32),
             inv_mass: wp.array(dtype=wp.float32),
             inertia: wp.array(dtype=wp.mat33),
             inv_inertia: wp.array(dtype=wp.mat33)):
    inv_mass[0] = 1.0/mass[0]
    hx = box_size[0]
    hy = box_size[1]
    hz = box_size[2]
    inertia[0] = wp.mat33(
        1.0/12.0*mass[0]*(hy*hy+hz*hz), 0.0, 0.0,
        0.0, 1.0/12.0*mass[0]*(hx*hx+hz*hz), 0.0,
        0.0, 0.0, 1.0/12.0*mass[0]*(hx*hx+hy*hy))
    inv_inertia[0] = wp.mat33(
        12.0/(mass[0]*(hy*hy+hz*hz)), 0.0, 0.0,
        0.0, 12.0/(mass[0]*(hx*hx+hz*hz)), 0.0,
        0.0, 0.0, 12.0/(mass[0]*(hx*hx+hy*hy)))


def quat_multiply(a, b):

    return np.array((a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - b[1] * a[2],
                     a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - b[2] * a[0],
                     a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - b[0] * a[1],
                     a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]))


class CubeSlopeSim:

    def __init__(self, seed=0, device='cpu'):

        self.seed = seed
        self.device = device

        self.frame_dt = 1.0/60.0

        self.episode_duration = 1.5      # seconds
        self.episode_frames = int(self.episode_duration/self.frame_dt)

        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_steps = int(self.episode_duration / self.sim_dt)

        self.sim_time = 0.0
        self.render_time = 0.0
        self.render = False

        self.states = []

    def forward(self, inputs, compute_grad=True):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print("inputs:", inputs.detach().cpu().numpy())

        self.num_envs = inputs.shape[0]

        builder = wp.sim.ModelBuilder()

        # rot_angles = torch.FloatTensor(
        #     self.num_envs, 3).uniform_(-math.pi, math.pi)
        rot_angles = torch.tensor([0.5, -0.4, 0.3]).tile(self.num_envs, 1)

        hx = 1.5
        hy = 2
        hz = 1

        V = 8*hx*hy*hz

        pos_low = (10+hx)/math.sqrt(3) + hy
        pos_high = 12.0

        pos = torch.FloatTensor(self.num_envs, 1).uniform_(pos_low, pos_high)

        for e in range(self.num_envs):
            density = 10.0/V

            rot_x = wp.quat_from_axis_angle((1.0, 0.0, 0.0), rot_angles[e][0])
            rot_y = wp.quat_from_axis_angle((0.0, 1.0, 0.0), rot_angles[e][1])
            rot_z = wp.quat_from_axis_angle((0.0, 0.0, 1.0), rot_angles[e][2])
            rot = quat_multiply(quat_multiply(rot_x, rot_y), rot_z)

            body_id = builder.add_body(
                origin=wp.transform((-10.0, 5.0, 0.0), rot))
            builder.add_shape_box(
                body=body_id,
                hx=hx,
                hy=hy,
                hz=hz,
                density=density,
                ke=2.e+3,
                kd=0.0,
                mu=0.2,
                restitution=0.0)
            # builder.add_shape_sphere(
            #     body=body_id,
            #     radius=hx,
            #     density=density,
            #     ke=2.e+3,
            #     kd=0.0,
            #     mu=0.2,
            #     restitution=0.0)

        # finalize model
        self.model = builder.finalize(self.device)
        self.model.ground = True

        self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.integrator = wp.sim.XPBDIntegrator(iterations=1, contact_con_weighting=False)
        self.loss = wp.zeros(self.num_envs*3, dtype=wp.float32,
                             device=self.device, requires_grad=True)

        # TODO avoid these reallocations during normal forward pass
        # if compute_grad or len(self.states) == 0:
        #     self.states = []
        #     for t in trange(self.episode_frames*self.sim_substeps+1, desc="Allocating states"):
        #         if t > 0 and not compute_grad:
        #             self.states.append(self.states[0])
        #         else:
        #             self.states.append(self.model.state(
        #                 requires_grad=compute_grad, require_contact_grads=compute_grad))
        # else:
        #     self.states[0] = self.model.state(
        #         requires_grad=False, require_contact_grads=False)
        self.states = []
        for t in trange(self.episode_frames*self.sim_substeps+1, desc="Allocating states"):
            self.states.append(self.model.state(
                requires_grad=compute_grad, require_contact_grads=compute_grad))

        # XXX set parameters here without applying any operations until we tape the simulation
        self.model.body_mass = wp.array(inputs.detach().cpu().numpy(
        ).flatten(), dtype=wp.float32, device=self.device, requires_grad=True)
        # self.model.body_inv_mass = wp.array(inputs.detach().cpu().numpy(
        # ).flatten(), dtype=wp.float32, device=self.device, requires_grad=compute_grad)

        self.model.body_mass.requires_grad = compute_grad
        self.model.body_inv_mass.requires_grad = compute_grad
        self.model.body_inertia.requires_grad = compute_grad
        self.model.body_inv_inertia.requires_grad = compute_grad
        self.model.body_com.requires_grad = compute_grad


        self.model.body_q.requires_grad = compute_grad
        self.model.body_qd.requires_grad = compute_grad

        if (self.model.ground):
            self.model.collide(self.states[0])

        self.model.ground_contact_ref.requires_grad = compute_grad
        self.model.ground_plane.requires_grad = compute_grad
        self.model.gravity.requires_grad = compute_grad

        self.model.shape_transform.requires_grad = compute_grad
        self.model.shape_contact_thickness.requires_grad = compute_grad

        self.model.shape_materials.ke.requires_grad = compute_grad
        self.model.shape_materials.kd.requires_grad = compute_grad
        self.model.shape_materials.kf.requires_grad = compute_grad
        self.model.shape_materials.mu.requires_grad = compute_grad
        self.model.shape_materials.restitution.requires_grad = compute_grad

        box_size = wp.array([hx*2, hy*2, hz*2], dtype=wp.float32, device=self.device)
        box_size.requires_grad = compute_grad

        self.joint_score = 0.0
        # -----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__),
                                                                               "outputs/example_sim_cube_slope.usd"))

        # ---------------
        # run simulation

        self.sim_time = 0.0

        num_obs = 1

        profiler = {}

        # create update graph
#        wp.capture_begin()
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):

            if (self.render):

                with wp.ScopedTimer("render", False):

                    if (self.render):
                        self.render_time += self.frame_dt

                        self.renderer.begin_frame(self.render_time)
                        self.renderer.render(self.states[0])
                        self.renderer.end_frame()

                self.renderer.save()

            tape = wp.Tape()
            with tape:
                wp.launch(set_mass,
                          dim=1,
                          inputs=[
                              self.model.body_mass,
                              box_size,
                          ],
                          outputs=[
                              self.model.body_inv_mass,
                              self.model.body_inertia,
                              self.model.body_inv_inertia,
                          ],
                          device=self.device)

                # simulate
                t = 0
                for f in trange(self.episode_frames, desc="Simulating"):
                    for i in range(self.sim_substeps):
                        self.model.allocate_rigid_contacts()
                        wp.sim.collide(self.model, self.states[t])

                        self.integrator.simulate(
                            self.model, self.states[t], self.states[t+1], self.sim_dt)
                        self.sim_time += self.sim_dt
                        t += 1

                    if (self.render):

                        with wp.ScopedTimer("render", False):
                            if (self.render):
                                self.render_time += self.frame_dt
                                self.renderer.begin_frame(self.render_time)
                                self.renderer.render(self.states[t])
                                self.renderer.end_frame()

                if (self.render):
                    self.renderer.save()
                    import sys
                    sys.exit()

                wp.launch(loss_kernel, dim=(1, 3), inputs=[
                          self.states[-1].body_q, self.loss], device=self.device)
                print("loss:", self.loss.numpy())

            if compute_grad:
                tape.backward(grads={self.loss: wp.array(
                    [0, 1, 0]*self.num_envs, dtype=wp.float32, device=self.device)})
                grad_mass = wp.to_torch(
                    tape.gradients[self.model.body_mass]).clone()
                tape.zero()
            else:
                grad_mass = None

            wp.synchronize()

        return wp.to_torch(self.loss)[1], grad_mass


seed = 1

np.set_printoptions(precision=10)

# robot = CubeSlopeSim(seed=seed, device=wp.get_preferred_device())
robot = CubeSlopeSim(seed=seed, device="cpu")
torch.manual_seed(seed)
# param = torch.tensor([1 / 10.0]).repeat(1, 1).view(1, 1)
param = torch.tensor([100.0]).repeat(1, 1).view(1, 1)

# robot.render=True
# robot.forward(param, compute_grad=False)
# import sys
# sys.exit(0)


_, autodiff_mass = robot.forward(param)
print("autodiff:", autodiff_mass)


def f(x):
    return robot.forward(torch.tensor(x).view(1, 1), compute_grad=False)[0].item()


print("finite difference:", derivative(f, param.item(), dx=1e-4, order=7))
# central difference
# eps = 1e-5
eps = 1e-4
print("central difference:", (f(param.item() + eps) - f(param.item() - eps)) / (2 * eps))
