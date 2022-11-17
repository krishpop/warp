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
# using the simulation gradients.
#
###########################################################################


import os

import numpy as np

import warp as wp
# wp.config.mode = "debug"
import warp.sim
import warp.sim.render
from warp.tests.grad_utils import *
from warp.optim import Adam

import matplotlib.pyplot as plt

from tqdm import trange

# wp.config.verify_fp = True
wp.init()

@wp.kernel
def loss_l2(states: wp.array(dtype=wp.float32), targets: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    diff = states[i] - targets[i]
    l = diff * diff
    wp.atomic_add(loss, 0, l)

@wp.kernel
def apply_torque(torques: wp.array(dtype=wp.float32), start_index: int, body_f: wp.array(dtype=wp.spatial_vector)):
    fx = torques[start_index + 0]
    fz = torques[start_index + 1]
    body_f[0] = wp.spatial_vector(0.0, 0.0, 0.0, fx, 0.0, fz)

@wp.kernel
def save_state(body_q: wp.array(dtype=wp.transform), start_index: int, states: wp.array(dtype=wp.float32)):
    pos = wp.transform_get_translation(body_q[0])
    states[start_index + 0] = pos[0]
    states[start_index + 1] = pos[2]

class Environment:

    frame_dt = 1.0/60.0

    episode_duration = 1.5      # seconds
    episode_frames = 100  # int(episode_duration/frame_dt)

    sim_substeps = 1
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    def __init__(self, device='cpu'):

        builder = wp.sim.ModelBuilder()

        self.device = device

        # add planar joints
        builder = wp.sim.ModelBuilder(gravity=0.0)
        builder.add_articulation()
        b = builder.add_body(
                parent=-1,
                origin=wp.transform())
        s = builder.add_shape_box( 
                pos=(0.0, 0.0, 0.0),
                hx=0.5,
                hy=0.5,
                hz=0.5,
                density=100.0,
                body=b)

        # compute reference trajectory
        rad = np.linspace(0.0, np.pi*2, self.episode_frames)
        self.ref_traj = np.stack([np.cos(rad), np.sin(rad)], axis=1)

        # set initial joint configuration to first reference state
        builder.body_q[0] = wp.transform(p=[self.ref_traj[0][0], 0.0, self.ref_traj[0][1]])

        builder.ground = False
        # plt.plot(self.ref_traj[:,0], self.ref_traj[:,1])
        # plt.show()

        self.ref_traj = wp.array(self.ref_traj.flatten(), dtype=wp.float32, device=self.device, requires_grad=True)

        # finalize model
        self.model = builder.finalize(device, requires_grad=True)

        self.model.joint_attach_kd = 0.0
        self.model.joint_limit_ke.zero_()
        self.model.joint_limit_kd.zero_()

        self.builder = builder
        self.model.ground = False

        self.dof_q = self.model.joint_coord_count
        self.dof_qd = self.model.joint_dof_count
        self.num_bodies = self.model.body_count

        self.action_dim = 2
        self.state_dim = 2

        assert len(self.ref_traj) == self.episode_frames * self.state_dim

        if False:
            # XXX apply initial generalized coordinates to the model
            # (subsequent state constructions via `model.state()` will
            # use these transforms)
            state = self.model.state()
            wp.sim.eval_fk(
                self.model,
                self.model.joint_q,
                self.model.joint_qd,
                None,
                state)
            self.model.body_q.assign(state.body_q)
            self.model.body_qd.assign(state.body_qd)

        # solve_iterations = 1
        # self.integrator = wp.sim.XPBDIntegrator(solve_iterations, contact_con_weighting=True)
        self.integrator = wp.sim.SemiImplicitIntegrator()

    def simulate(self, state: wp.sim.State, action: wp.array, action_index: int, requires_grad=False) -> wp.sim.State:
        """
        Simulate the system for the given states.
        """
        
        for _ in range(self.sim_substeps):
            if requires_grad:
                next_state = self.model.state(requires_grad=True)
            else:
                next_state = state
                next_state.clear_forces()
            # apply generalized torques to rigid body here, instead of planar joints
            wp.launch(
                apply_torque,
                1,
                inputs=[action, action_index],
                outputs=[state.body_f],
                device=action.device)
            # if self.model.ground:
            #     self.model.allocate_rigid_contacts(requires_grad=requires_grad)
            #     wp.sim.collide(self.model, state)
            state = self.integrator.simulate(self.model, state, next_state, self.sim_dt, requires_grad=requires_grad)
        return state

    def _render(self, state: wp.sim.State):
        self.renderer.begin_frame(self.render_time)
        self.renderer.render(state)
        self.renderer.end_frame()
        self.render_time += self.frame_dt
        self.renderer.save()

    def forward(self, actions: wp.array, requires_grad=False, loss=None, render=False):
        """
        Advances the system dynamics given the rigid-body state in maximal coordinates and generalized joint torques [body_q, body_qd, tau].
        Simulates for the set number of substeps and returns the next state in maximal and (optional) generalized coordinates [body_q_next, body_qd_next, joint_q_next, joint_qd_next].
        """

        actions.requires_grad = requires_grad
        state = self.model.state(requires_grad=requires_grad)
        if (render):
            # set up Usd renderer
            self.renderer = wp.sim.render.SimRenderer(
                self.model,
                os.path.join(os.path.dirname(__file__), "outputs/example_sim_trajopt.usd"),
                scaling=100.0)
            self.render_time = 0.0
            self._render(state)

        states = wp.zeros(self.episode_frames * self.state_dim, dtype=wp.float32, device=self.device, requires_grad=requires_grad)

        for i in range(self.episode_frames):

            # simulate
            next_state = self.simulate(state, actions, action_index=i*self.action_dim, requires_grad=requires_grad)

            # save state
            wp.launch(
                save_state,
                dim=1,
                inputs=[next_state.body_q, i*self.state_dim],
                outputs=[states],
                device=self.device)

            # update state
            state = next_state

            if (render):
                self._render(state)

        if False:
            np_states = states.numpy().reshape((-1, 2))
            print("states", np_states)
            np_ref = self.ref_traj.numpy().reshape((-1, 2))
            plt.plot(np_ref[:,0], np_ref[:,1], label="ref")
            plt.plot(np_states[:,0], np_states[:,1], label="states")
            plt.grid()
            plt.legend()
            plt.axis('equal')
            plt.show()

        # compute loss
        if loss is None:
            loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        wp.launch(
            loss_l2,
            dim=self.state_dim*self.episode_frames,
            inputs=[states, self.ref_traj],
            outputs=[loss],
            device=self.device)
        
        return states

    def optimize(self, num_iter=100, lr=0.01):
        # initial guess
        actions = wp.array(np.ones(self.episode_frames*self.action_dim)*100.0, dtype=wp.float32, device=self.device, requires_grad=True)

        optimizer = Adam([actions], lr=lr)
        loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        progress = trange(num_iter, desc="Optimizing")

        # optimize
        for i in progress:
            loss.zero_()
            tape = wp.Tape()
            with tape:
                self.forward(actions, requires_grad=True, loss=loss)

            # check_backward_pass(
            #     tape,
            #     plot_jac_on_fail=True,
            #     track_inputs=[actions],
            #     track_outputs=[loss],
            #     track_input_names=["actions"],
            #     track_output_names=["loss"],)

            progress.set_description(f"Optimizing, loss: {loss.numpy()[0]:.3f}")

            tape.backward(loss=loss)
            # print("action grad", actions.grad.numpy())
            assert not np.isnan(actions.grad.numpy()).any(), "NaN in gradient"
            optimizer.step([actions.grad])
            tape.zero()

        return actions

np.set_printoptions(precision=4, linewidth=2000, suppress=True)

sim = Environment(device=wp.get_preferred_device())

best_actions = sim.optimize(num_iter=250, lr=1e3)
# print("best actions", best_actions.numpy())
# render
opt_traj = sim.forward(best_actions, render=True)

np_states = opt_traj.numpy().reshape((-1, 2))
np_ref = sim.ref_traj.numpy().reshape((-1, 2))
plt.plot(np_ref[:,0], np_ref[:,1], label="reference")
plt.plot(np_states[:,0], np_states[:,1], label="optimized")
plt.grid()
plt.legend()
plt.axis('equal')
plt.show()

import sys
sys.exit(0)
