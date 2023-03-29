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
wp.config.mode = "debug"
import warp.sim
import warp.sim.render


wp.config.verify_fp = False
wp.config.verbose = False
wp.init()

@wp.kernel
def loss_kernel(body_q: wp.array(dtype=wp.transform), loss: wp.array(dtype=float, ndim=1)):
    tid = wp.tid()
    i = tid//3
    j = tid % 3
    x = wp.transform_get_translation(body_q[i])
    loss[i*3+j] = x[j]


@wp.kernel
def set_mass(in_mass: wp.array(dtype=wp.float32),
             box_size: wp.array(dtype=wp.float32),
             out_mass: wp.array(dtype=wp.float32),
             inv_mass: wp.array(dtype=wp.float32),
             inertia: wp.array(dtype=wp.mat33),
             inv_inertia: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    out_mass[tid] = in_mass[tid]
    inv_mass[tid] = 1.0/in_mass[tid]
    hx = box_size[0]
    hy = box_size[1]
    hz = box_size[2]
    inertia[tid] = wp.mat33(
        1.0/12.0*in_mass[tid]*(hy*hy+hz*hz), 0.0, 0.0,
        0.0, 1.0/12.0*in_mass[tid]*(hx*hx+hz*hz), 0.0,
        0.0, 0.0, 1.0/12.0*in_mass[tid]*(hx*hx+hy*hy))
    inv_inertia[tid] = wp.mat33(
        12.0/(in_mass[tid]*(hy*hy+hz*hz)), 0.0, 0.0,
        0.0, 12.0/(in_mass[tid]*(hx*hx+hz*hz)), 0.0,
        0.0, 0.0, 12.0/(in_mass[tid]*(hx*hx+hy*hy)))


def quat_multiply(a, b):

    return np.array((a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - b[1] * a[2],
                     a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - b[2] * a[0],
                     a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - b[0] * a[1],
                     a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]))


class CubeSlopeSim:

    def __init__(self, num_envs=1, seed=0, device='cpu'):

        self.seed = seed
        self.device = device

        self.frame_dt = 1.0/60.0

        self.episode_duration = 2.5      # seconds
        self.episode_frames = int(self.episode_duration/self.frame_dt)

        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_steps = int(self.episode_duration / self.sim_dt)

        self.sim_time = 0.0
        self.render_time = 0.0
        self.render = False

        self.num_envs = num_envs
        self.loss = wp.zeros(self.num_envs*3, dtype=wp.float32,
                             device=self.device, requires_grad=True)
        self.loss_test_dim = 0

        
        builder = wp.sim.ModelBuilder()

        # rot_angles = torch.FloatTensor(
        #     self.num_envs, 3).uniform_(-math.pi, math.pi)
        rot_angles = torch.tensor([0.3, -0.4, 0.3]).tile(self.num_envs, 1)

        hx = 1.5
        hy = 2
        hz = 1
        self.hx = hx
        self.hy = hy
        self.hz = hz

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
                ke=2.e+4,
                kd=0.0,
                mu=0.2,
                restitution=0.0)
            # builder.add_shape_sphere(
            #     body=body_id,
            #     radius=hx,
            #     density=density,
            #     ke=2.e+3,
            #     kd=0.0,
            #     mu=0.5,
            #     restitution=0.5)
            
            # apply some initial velocity
            builder.body_qd[-6:] = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0]

        self.builder = builder

        self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.integrator = wp.sim.XPBDIntegrator(iterations=1, rigid_contact_con_weighting=False)


    def simulate(self, input_mass: wp.array, compute_grad=True):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        model = self.builder.finalize(self.device, requires_grad=compute_grad) 
        model.ground = True
        state = model.state(requires_grad=compute_grad)
        if (model.ground):
            model.collide(state)

        box_size = wp.array([self.hx*2, self.hy*2, self.hz*2], dtype=wp.float32, device=self.device)
        box_size.requires_grad = compute_grad

        self.joint_score = 0.0
        # -----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(
                model, os.path.join(os.path.dirname(__file__),
                "outputs/example_sim_cube_slope.usd"))

        # ---------------
        # run simulation

        self.sim_time = 0.0
        
        wp.launch(
            set_mass,
            dim=self.num_envs,
            inputs=[
                input_mass,
                box_size,
            ],
            outputs=[
                model.body_mass,
                model.body_inv_mass,
                model.body_inertia,
                model.body_inv_inertia,
            ],
            device=self.device)

        # simulate
        for f in trange(self.episode_frames, desc="Simulating"):
            for i in range(self.sim_substeps):
                model.allocate_rigid_contacts(requires_grad=compute_grad)
                if compute_grad:
                    next_state = model.state(requires_grad=True)
                else:
                    # we can simply overwrite the state
                    next_state = state
                next_state.clear_forces()
                wp.sim.collide(model, state)

                state = self.integrator.simulate(
                    model, state, next_state, self.sim_dt, requires_grad=compute_grad)
                self.sim_time += self.sim_dt

            if (self.render):

                with wp.ScopedTimer("render", False):
                    if (self.render):
                        self.render_time += self.frame_dt
                        self.renderer.begin_frame(self.render_time)
                        self.renderer.render(state)
                        self.renderer.end_frame()

        if (self.render):
            self.renderer.save()
            import sys
            sys.exit()

        wp.launch(loss_kernel,
            dim=(1, 3),
            inputs=[state.body_q],
            outputs=[self.loss],
            device=self.device)
        return self.loss

    def forward(self, inputs, compute_grad=True):
        wp_param = wp.array(inputs.detach().cpu().numpy().flatten(), dtype=wp.float32, device=self.device)
        wp_param.requires_grad = True

        tape = wp.Tape()
        with tape:
            self.simulate(wp_param, compute_grad=compute_grad)

        if compute_grad:
            test_vec = np.zeros(3, dtype=np.float32)
            # test_vec = np.ones(3, dtype=np.float32)
            test_vec[self.loss_test_dim] = 1.0
            tape.backward(grads={self.loss: wp.array(
                np.tile(test_vec, self.num_envs), dtype=wp.float32, device=self.device)})
            grad_mass = wp.to_torch(
                tape.gradients[wp_param]).clone()
            tape.zero()
        else:
            grad_mass = None

        wp.synchronize()

        return wp.to_torch(self.loss)[self.loss_test_dim], grad_mass


seed = 1

np.set_printoptions(precision=10)

robot = CubeSlopeSim(seed=seed, device=wp.get_preferred_device())
# robot = CubeSlopeSim(seed=seed, device="cpu")
torch.manual_seed(seed)
# param = torch.tensor([1 / 10.0]).repeat(1, 1).view(1, 1)
param = torch.tensor([100.0]).repeat(1, 1).view(1, 1)

# robot.render = True
# robot.forward(param, compute_grad=True)
# import sys
# sys.exit(0)

# wp_param = wp.array(param.detach().cpu().numpy().flatten(), dtype=wp.float32, device=robot.device)
# wp_param.requires_grad = True
# from warp.tests.grad_utils import check_backward_pass
# tape = wp.Tape()
# with tape:
#     robot.simulate(wp_param, compute_grad=True)
# check_backward_pass(tape,
#     visualize_graph=False,
#     check_jacobians=True,
#     plot_jac_on_fail=True,
#     track_inputs=[wp_param], 
#     track_outputs=[robot.loss],
#     ignore_kernels={
#         "update_rigid_ground_contacts",
#         "set_mass",
#         "loss_kernel",
#         "integrate_bodies",
#         "eval_body_joints"
#     })
# import sys
# sys.exit(0)


_, autodiff_grad = robot.forward(param, compute_grad=True)
print("autodiff:", autodiff_grad)


def f(x):
    return robot.forward(torch.tensor(x).view(1, 1), compute_grad=False)[0].item()


print("finite difference:", derivative(f, param.item(), dx=1e-4, order=7))
# central difference
# eps = 1e-5
eps = 1e-4
print("central difference:", (f(param.item() + eps) - f(param.item() - eps)) / (2 * eps))
