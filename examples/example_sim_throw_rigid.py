# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Trajectory Optimization
#
# Shows how to optimize torque trajectories for a simple planar environment
# using Warp's provided Adam optimizer.
#
###########################################################################


import os

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
from warp.optim import Adam

import matplotlib.pyplot as plt

from warp.tests.grad_utils import *

wp.init()


@wp.kernel
def sim_loss(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector), target_pos: wp.vec3, loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    tf = body_q[i]
    dist = wp.length_sq(wp.transform_get_translation(tf) - target_pos)
    vel = wp.length_sq(body_qd[i])
    wp.atomic_add(loss, 0, dist + 0.1 * vel)


@wp.kernel
def apply_velocity(action: wp.array(dtype=wp.vec3), body_qd: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    # apply linear velocity
    body_qd[i] = wp.spatial_vector(wp.vec3(0.0), action[i])


class Environment:
    frame_dt = 1.0 / 60.0
    episode_frames = 100

    sim_substeps = 3
    sim_dt = frame_dt / sim_substeps

    sim_time = 0.0
    render_time = 0.0

    def __init__(self, device="cpu"):
        builder = wp.sim.ModelBuilder()

        self.device = device

        self.start_pos = wp.vec3(0.0, 1.6, 0.0)
        self.target_pos = wp.vec3(3.0, 0.6, 0.0)

        # add planar joints
        builder = wp.sim.ModelBuilder(gravity=0.0)
        builder.add_articulation()
        b = builder.add_body(origin=wp.transform(self.start_pos))
        s = builder.add_shape_box(pos=(0.0, 0.0, 0.0), hx=0.5, hy=0.5, hz=0.5, density=100.0, body=b)

        # finalize model
        self.model = builder.finalize(device, requires_grad=True)

        self.builder = builder
        self.model.ground = True

        solve_iterations = 2
        self.integrator = wp.sim.XPBDIntegrator(solve_iterations)
        # self.integrator = wp.sim.SemiImplicitIntegrator()

        self.num_iterations = 100

    def simulate(self, state: wp.sim.State, action: wp.array, requires_grad=False) -> wp.sim.State:
        """
        Simulate the system for the given states.
        """

        self.render_time = 0.0
        traj_verts = [state.body_q.numpy()[0, :3].tolist()]
        for frame in range(self.episode_frames):
            for i in range(self.sim_substeps):
                if requires_grad:
                    next_state = self.model.state(requires_grad=True)
                else:
                    next_state = state
                    next_state.clear_forces()

                if i == 0:
                    # apply initial velocity to the rigid object
                    wp.launch(apply_velocity, 1, inputs=[action], outputs=[state.body_qd], device=action.device)

                self.model.allocate_rigid_contacts(requires_grad=True)
                wp.sim.collide(self.model, state)
                self.integrator.simulate(self.model, state, next_state, self.sim_dt, requires_grad=requires_grad)

                if self.renderer is not None:
                    self.renderer.begin_frame(self.render_time)
                    self.renderer.render(state)
                    self.renderer.end_frame()
                    self.render_time += self.frame_dt
                    traj_verts.append(next_state.body_q.numpy()[0, :3].tolist())
                    self.renderer.render_line_strip(
                        vertices=traj_verts,
                        color=wp.render.bourke_color_map(0.0, self.num_iterations, self.iteration),
                        radius=0.02 + 0.01 * self.iteration / self.num_iterations,
                        name=f"traj_{self.iteration}",
                    )
                    # self.renderer.save()

                state = next_state

        return state

    def optimize(self, num_iter=100, lr=0.01, render=True):

        action = wp.zeros(1, dtype=wp.vec3, requires_grad=True)

        optimizer = Adam([action], lr=lr)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        self.num_iterations = num_iter

        if render:
            # set up Usd renderer
            self.renderer = wp.sim.render.SimRendererOpenGL(
                self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_trajopt.usd"), scaling=1.0
            )
            self.renderer.render_sphere("target", self.target_pos, wp.quat_identity(), 0.1)
        else:
            self.renderer = None

        # optimize
        losses = []
        for i in range(1, num_iter + 1):
            self.iteration = i
            loss.zero_()
            state = self.model.state(requires_grad=False)
            tape = wp.Tape()
            with tape:
                final_state = self.simulate(state, action, requires_grad=True)

                wp.launch(sim_loss, dim=1, inputs=[final_state.body_q, final_state.body_qd, self.target_pos], outputs=[loss], device=action.device)

            # check_backward_pass(tape, visualize_graph=False,)
            l = loss.numpy()[0]
            print(f"iter {i}/{num_iter} loss: {l:.3f}")
            losses.append(l)

            tape.backward(loss=loss)
            # print("action grad", opt_vars.grad.numpy())
            assert not np.isnan(action.grad.numpy()).any(), "NaN in gradient"
            optimizer.step([action.grad])
            tape.zero()

        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.grid()
        plt.title("Loss")
        plt.xlabel("Iteration")
        plt.show()

        return action


np.set_printoptions(precision=4, linewidth=2000, suppress=True)

sim = Environment(device=wp.get_preferred_device())

best_actions = sim.optimize(num_iter=50, lr=1e-1)

# np_states = opt_traj.numpy().reshape((-1, 2))
# np_ref = sim.ref_traj.numpy().reshape((-1, 2))
# plt.plot(np_ref[:, 0], np_ref[:, 1], label="reference")
# plt.plot(np_states[:, 0], np_states[:, 1], label="optimized")
# plt.grid()
# plt.legend()
# plt.axis("equal")
# plt.show()
