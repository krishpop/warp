# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Grad Bounce
#
# Shows how to use Warp to optimize the initial velocity of a particle
# such that it bounces off the wall and floor in order to hit a target.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
###########################################################################

import os

import numpy as np
import warp as wp

import warp.sim
import warp.sim.render
import pdb

from warp.tests.grad_utils import check_backward_pass, check_tape_safety

wp.init()


''' From older git commit (Eric's repo) before mesh gradient fixes'''


class Bounce:

    # seconds
    sim_duration = 0.2

    # control frequency
    frame_dt = 1.0 / 60.0
    frame_steps = int(sim_duration / frame_dt)

    # sim frequency
    sim_substeps = 5  # 8
    sim_steps = frame_steps * sim_substeps
    sim_dt = frame_dt / sim_substeps
    sim_time = 0.0

    render_time = 0.0

    train_iters = 250
    train_rate = 0.01

    def __init__(self, render=True, profile=False, adapter=None):

        builder = wp.sim.ModelBuilder()

        self.contact_thickness = 1e-3

        # builder.add_particle(pos=(-0.5, 1.0, 0.0), vel=(5.0, -5.0, 0.0), mass=1.0)
        builder.add_particle(pos=(0., 0.1 + self.contact_thickness, 0.0), vel=(0.0, 0.0, 0.), mass=1.0)
        # builder.add_shape_box(body=-1, pos=(2.0, 1.0, 0.0), hx=0.25, hy=1.0, hz=1.0)

        self.device = wp.get_device(adapter)
        self.profile = profile

        self.model = builder.finalize(self.device, requires_grad=True)
        self.model.ground = True

        self.model.soft_contact_ke = 1.e+4
        self.model.soft_contact_kf = 0.0
        self.model.soft_contact_kd = 1.e+1
        self.model.soft_contact_mu = 0.25
        self.model.soft_contact_margin = 10.0

        if True:
            self.model.particle_radius = wp.array([0.04], dtype=float, device=self.device, requires_grad=True)
            self.model.soft_contact_restitution = 0.5

            # particle-body and particle-ground contact
            self.model.soft_contact_ke = 500  # 1.e+4
            self.model.soft_contact_kf = 300  # 1000
            self.model.soft_contact_kd = 10  # 100
            self.model.soft_contact_mu = 0.2  # 0.75

            self.model.soft_contact_margin = 100.0
            self.model.particle_cohesion = 0.05
            self.model.particle_adhesion = 0.0
            self.model.soft_contact_distance = 0.05

            xpbd_settings = dict(iterations=3,  # 10, #50,
                                 joint_linear_relaxation=1.0,
                                 joint_angular_relaxation=0.45,
                                 rigid_contact_relaxation=1.0,
                                 rigid_contact_con_weighting=True,
                                 enable_restitution=True,
                                 )

        self.integrator = wp.sim.XPBDIntegrator(**xpbd_settings)
        # self.integrator = wp.sim.SemiImplicitIntegrator()

        # self.integrator = wp.sim.XPBDIntegrator() #SemiImplicitIntegrator()

        self.target = (-2.0, 1.5, 0.0)
        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for i in range(self.sim_steps + 1):
            self.states.append(self.model.state(requires_grad=True))

        # one-shot contact creation (valid if we're doing simple collision against a constant normal plane)
        # wp.sim.collide(self.model, self.states[0])

        if (self.render):
            self.stage = wp.sim.render.SimRenderer(
                self.model,
                os.path.join(os.path.dirname(__file__), "outputs/example_sim_grad_bounce.usd"),
                scaling=40.0)

    @wp.kernel
    def loss_kernel(pos: wp.array(dtype=wp.vec3),
                    target: wp.vec3,
                    loss: wp.array(dtype=float)):

        # distance to target
        delta = pos[0] - target
        loss[0] = wp.dot(delta, delta)

    @ wp.kernel
    def state_loss_kernel(pos: wp.array(dtype=wp.vec3),
                          loss: wp.array(dtype=float)):
        # loss[0] = pos[0][0] + pos[0][1] + pos[0][2]
        loss[0] = wp.length_sq(pos[0])

    @wp.kernel
    def step_kernel(x: wp.array(dtype=wp.vec3),
                    grad: wp.array(dtype=wp.vec3),
                    alpha: float):

        tid = wp.tid()

        # gradient descent step
        x[tid] = x[tid] - grad[tid] * alpha

    def compute_loss(self):

        # run control loop
        for i in range(self.sim_steps):

            self.states[i].clear_forces()

            wp.sim.collide(self.model, self.states[i])

            self.integrator.simulate(self.model,
                                     self.states[i],
                                     self.states[i + 1],
                                     self.sim_dt,
                                     requires_grad=True)

            # print("step", i, self.states[i].particle_q.numpy())

        # compute loss on final state
        # wp.launch(self.loss_kernel, dim=1, inputs=[self.states[-1].particle_q, self.target, self.loss], device=self.device)
        wp.launch(self.state_loss_kernel, dim=1, inputs=[self.states[-1].particle_q], outputs=[self.loss], device=self.device)

        return self.loss

    def render(self, iter):

        # render every 16 iters
        if iter % 16 > 0:
            return

        # draw trajectory
        traj_verts = [self.states[0].particle_q.numpy()[0].tolist()]

        for i in range(0, self.sim_steps, self.sim_substeps):

            traj_verts.append(self.states[i].particle_q.numpy()[0].tolist())

            self.stage.begin_frame(self.render_time)
            self.stage.render(self.states[i])
            self.stage.render_box(pos=self.target, rot=wp.quat_identity(), extents=(0.1, 0.1, 0.1), name="target")
            self.stage.render_line_strip(vertices=traj_verts, color=wp.render.bourke_color_map(
                0.0, 7.0, self.loss.numpy()[0]), radius=0.02, name=f"traj_{iter}")
            self.stage.end_frame()

            self.render_time += self.frame_dt

        self.stage.save()

    def check_grad(self):

        param = self.states[0].particle_q

        # initial value
        x_c = param.numpy().flatten().copy()

        # compute numeric gradient
        x_grad_numeric = np.zeros_like(x_c)

        for i in range(len(x_c)):

            eps = 1.e-3

            step = np.zeros_like(x_c)
            step[i] = eps

            x_1 = x_c + step
            x_0 = x_c - step

            param.assign(x_1)
            l_1 = self.compute_loss().numpy()[0]

            param.assign(x_0)
            l_0 = self.compute_loss().numpy()[0]

            dldx = (l_1 - l_0) / (eps * 2.0)

            x_grad_numeric[i] = dldx

        # reset initial state
        param.assign(x_c)

        # # compute analytic gradient
        # tape = wp.Tape()
        # with tape:
        #     l = self.compute_loss()

        # tape.backward(l)

        # x_grad_analytic = tape.gradients[param]

        # if False:
        #     print(f"numeric grad: {x_grad_numeric}")
        #     print(f"analytic grad: {x_grad_analytic}")

        # tape.zero()

        return x_grad_numeric

    def train(self):

        # check whether all operations inside compute_loss are recorded on the tape
        # check_tape_safety(self.compute_loss, inputs=[])

        for i in range(1):  # (self.train_iters):

            tape = wp.Tape()

            with wp.ScopedTimer("Forward", active=self.profile):
                with tape:
                    self.compute_loss()

            # check_backward_pass(
            #     tape,
            #     track_inputs=[self.states[0].particle_q],
            #     track_outputs=[self.loss],
            #     track_input_names=["q at t=0"],
            #     track_output_names=["loss"],
            #     analyze_graph=False,
            #     visualize_graph=False,
            #     check_kernel_jacobians=True,
            #     ignore_kernels=["integrate_particles"]
            # )

            with wp.ScopedTimer("Backward", active=self.profile):
                tape.backward(self.loss)

            with wp.ScopedTimer("Render", active=self.profile):
                self.render(i)

            print(
                f"Analytic grad of particle_q at t+1 wrt itself: {tape.gradients[self.states[-1].particle_q].numpy()}")
            print(
                f"Analytic grad of particle_q at t+1 wrt particle_q at t: {tape.gradients[self.states[-2].particle_q].numpy()}")
            print(
                f"Analytic grad of particle_q at t+1 wrt particle_qd at t: {tape.gradients[self.states[-2].particle_qd].numpy()}")
            print(
                f"Analytic grad of particle_q at t+1 wrt particle_q at 0: {tape.gradients[self.states[0].particle_q].numpy()}")

            grad_numeric = self.check_grad()
            print(f"Numeric grad of particle_q at t+1 wrt particle_q at 0: {grad_numeric}")
            # with wp.ScopedTimer("Step", active=self.profile):
            #     x = self.states[0].particle_qd
            #     x_grad = tape.gradients[self.states[0].particle_qd]
            #     pdb.set_trace()

            #     print(f"Iter: {i} Loss: {self.loss}")
            #     print(f"   x: {x} g: {x_grad}")

            #     wp.launch(self.step_kernel, dim=len(x), inputs=[x, x_grad, self.train_rate], device=self.device)

            tape.zero()

    def train_graph(self):

        # capture forward/backward passes
        wp.capture_begin()

        tape = wp.Tape()
        with tape:
            self.compute_loss()

        tape.backward(self.loss)

        self.graph = wp.capture_end()

        # replay and optimize
        for i in range(1):  # self.train_iters):

            with wp.ScopedTimer("Step", active=self.profile):

                # forward + backward
                wp.capture_launch(self.graph)

                # gradient descent step
                x = self.states[0].particle_qd
                wp.launch(self.step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate], device=self.device)

                print(f"Iter: {i} Loss: {self.loss}")
                print(tape.gradients[self.states[0].particle_qd])

                # clear grads for next iteration
                tape.zero()

            with wp.ScopedTimer("Render", active=self.profile):
                self.render(i)


bounce = Bounce(profile=False, render=True)
# bounce.check_grad()
bounce.train()
# bounce.train_graph()
