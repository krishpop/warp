# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Control Gradient Computation
#
# Shows how to compute gradients of simulation states from a rigid-body
# simulation involving contacts.
#
###########################################################################


import os

import numpy as np

import warp as wp
wp.config.mode = "debug"
import warp.sim
import warp.sim.render
from warp.tests.grad_utils import *

from tqdm import trange

# wp.config.verify_fp = True
wp.init()

@wp.kernel
def inplace_assign(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    b[tid] = a[tid]

@wp.kernel
def inplace_assign_transform(a: wp.array(dtype=wp.transform), b: wp.array(dtype=wp.transform)):
    tid = wp.tid()
    b[tid] = a[tid]

@wp.kernel
def inplace_assign_spatial_vector(a: wp.array(dtype=wp.spatial_vector), b: wp.array(dtype=wp.spatial_vector)):
    tid = wp.tid()
    b[tid] = a[tid]

class RigidBodySimulator:
    """
    Differentiable simulator of a rigid-body system with contacts.
    The system state is described entirely by the joint positions q, velocities qd, and
    joint torques tau. The system state is updated by calling the warp_step() function.
    """

    frame_dt = 1.0/60.0

    episode_duration = 1.5      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 30
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    use_single_cartpole = True

    def __init__(self, render=False, num_envs=1, device='cpu'):

        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs

        for i in range(num_envs):
            if self.use_single_cartpole:
                wp.sim.parse_urdf(
                    os.path.join(os.path.dirname(__file__), "assets/cartpole_single.urdf"),
                    builder,
                    xform=wp.transform(
                        np.array((0.0, 1.0, 0.0)),
                        wp.quat_from_axis_angle((1.0, 0.0, 0.0), -np.pi*0.5)),
                    floating=False, 
                    density=0,
                    armature=0.1,
                    stiffness=0.0,
                    damping=0.0,
                    shape_ke=1.e+4,
                    shape_kd=0.0,
                    shape_kf=1.e+2,
                    shape_mu=1.0,
                    limit_ke=1.e+4,
                    limit_kd=0.0)
                # joint initial positions
                builder.joint_q[-2:] = [0.0, 0.3]
                builder.joint_target[:2] = [0.0, 0.0]
            else:
                wp.sim.parse_urdf(
                    os.path.join(os.path.dirname(__file__), "assets/cartpole.urdf"),
                    builder,
                    xform=wp.transform(
                        np.array((i*2.0, 4.0, 0.0)),
                        wp.quat_from_axis_angle((1.0, 0.0, 0.0), -np.pi*0.5)),
                    floating=False, 
                    density=0,
                    armature=0.1,
                    stiffness=0.0,
                    damping=0.0,
                    shape_ke=1.e+4,
                    shape_kd=1.e+2,
                    shape_kf=1.e+2,
                    shape_mu=1.0,
                    limit_ke=1.e+4,
                    limit_kd=1.e+1)

                builder.joint_q[-3:] = [0.0, 0.3, 0.0]
                builder.joint_target[:3] = [0.0, 0.0, 0.0]

            if False:
                self.chain_length = 1
                self.chain_width = 1.0
                self.chain_types = [
                    wp.sim.JOINT_REVOLUTE,
                    # wp.sim.JOINT_FREE,
                    # wp.sim.JOINT_FIXED, 
                    # wp.sim.JOINT_BALL,
                    # wp.sim.JOINT_UNIVERSAL,
                    # wp.sim.JOINT_COMPOUND
                    ]

                builder = wp.sim.ModelBuilder()

                for c, t in enumerate(self.chain_types):

                    # start a new articulation
                    builder.add_articulation()

                    for i in range(self.chain_length):

                        if i == 0:
                            parent = -1
                            parent_joint_xform = wp.transform([0.0, 0.0, c*1.0], wp.quat_identity())           
                        else:
                            parent = builder.joint_count-1
                            parent_joint_xform = wp.transform([self.chain_width, 0.0, 0.0], wp.quat_identity())

                        joint_type = t

                        if joint_type == wp.sim.JOINT_REVOLUTE:

                            joint_axis=(0.0, 0.0, 1.0)
                            joint_limit_lower=-np.deg2rad(60.0)
                            joint_limit_upper=np.deg2rad(60.0)

                        elif joint_type == wp.sim.JOINT_UNIVERSAL:
                            joint_axis=(1.0, 0.0, 0.0)
                            joint_limit_lower=-np.deg2rad(60.0),
                            joint_limit_upper=np.deg2rad(60.0),

                        elif joint_type == wp.sim.JOINT_BALL:
                            joint_axis=(0.0, 0.0, 0.0)
                            joint_limit_lower = 100.0
                            joint_limit_upper = -100.0

                        elif joint_type == wp.sim.JOINT_FIXED:
                            joint_axis=(0.0, 0.0, 0.0)
                            joint_limit_lower = 0.0
                            joint_limit_upper = 0.0
                    
                        elif joint_type == wp.sim.JOINT_COMPOUND:
                            joint_limit_lower=-np.deg2rad(60.0)
                            joint_limit_upper=np.deg2rad(60.0)

                        # create body
                        b = builder.add_body(
                                parent=parent,
                                origin=wp.transform([i, 0.0, c*1.0], wp.quat_identity()),
                                joint_xform=parent_joint_xform,
                                joint_axis=joint_axis,
                                joint_type=joint_type,
                                joint_limit_lower=joint_limit_lower,
                                joint_limit_upper=joint_limit_upper,
                                joint_target_ke=0.0,
                                joint_target_kd=0.0,
                                joint_limit_ke=30.0,
                                joint_limit_kd=30.0,
                                joint_armature=0.1)

                        # create shape
                        s = builder.add_shape_box( 
                                pos=(self.chain_width*0.5, 0.0, 0.0),
                                hx=self.chain_width*0.5,
                                hy=0.1,
                                hz=0.1,
                                density=10.0,
                                body=b)
        
        # create quaternions away from singularities so that finite differences can be used directly
        # axis = np.array([1.0, 2.0, 3.0])
        # axis /= np.linalg.norm(axis)
        # quat = wp.quat_from_axis_angle(axis, -math.pi*0.5)
        # builder.joint_X_p = [wp.transform((1.0, 2.0, 3.0), quat)]

        # finalize model
        self.model = builder.finalize(device, requires_grad=True)

        self.model.joint_attach_kd = 0.0
        self.model.joint_limit_ke.zero_()
        self.model.joint_limit_kd.zero_()

        self.builder = builder
        self.model.ground = False

        if self.use_single_cartpole:
            self.model.joint_attach_ke = 40000.0
            self.model.joint_attach_kd = 200.0
        else:
            self.model.joint_attach_ke = 1600.0
            self.model.joint_attach_kd = 20.0


        self.dof_q = self.model.joint_coord_count
        self.dof_qd = self.model.joint_dof_count
        self.num_bodies = self.model.body_count


        # XXX apply initial transforms to the model
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

        solve_iterations = 1
        # self.integrator = wp.sim.XPBDIntegrator(solve_iterations, contact_con_weighting=True)
        self.integrator = wp.sim.SemiImplicitIntegrator()

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(
                self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_grad_control.usd"))
        self.render_time = 0.0

    def simulate(self, state: wp.sim.State, requires_grad=False) -> wp.sim.State:
        """
        Simulate the system for the given states.
        """
        for _ in range(self.sim_substeps):
            if requires_grad:
                next_state = self.model.state(requires_grad=True)
            else:
                next_state = state
                next_state.clear_forces()
            if self.model.ground:
                self.model.allocate_rigid_contacts(requires_grad=requires_grad)
                wp.sim.collide(self.model, state)
            state = self.integrator.simulate(self.model, state, next_state, self.sim_dt, requires_grad=requires_grad)
        return state

    def _render(self, state: wp.sim.State):
        if (self.render):
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(state)
            self.renderer.end_frame()
            self.render_time += self.frame_dt
            self.renderer.save()

    def warp_step_maximal(self, body_q: wp.array, body_qd: wp.array, tau: wp.array, body_q_next: wp.array, body_qd_next: wp.array, joint_q_next: wp.array = None, joint_qd_next: wp.array = None, requires_grad=False, check_diffs=False):
        """
        Advances the system dynamics given the rigid-body state in maximal coordinates and generalized joint torques [body_q, body_qd, tau].
        Simulates for the set number of substeps and returns the next state in maximal and (optional) generalized coordinates [body_q_next, body_qd_next, joint_q_next, joint_qd_next].
        """

        if check_diffs:
            model_before = self.builder.finalize(self.device)

        start_state = self.model.state(requires_grad=requires_grad)

        # assign maximal state coordinates        
        wp.launch(inplace_assign_transform, dim=self.num_bodies, inputs=[body_q], outputs=[start_state.body_q], device=self.device)
        wp.launch(inplace_assign_spatial_vector, dim=self.num_bodies, inputs=[body_qd], outputs=[start_state.body_qd], device=self.device)

        # assign input controls as joint torques
        wp.launch(inplace_assign, dim=self.dof_qd, inputs=[tau], outputs=[self.model.joint_act], device=self.device)
        
        end_state = self.simulate(start_state, requires_grad=requires_grad)

        if joint_q_next is not None and joint_qd_next is not None:
            wp.sim.eval_ik(self.model, end_state, joint_q_next, joint_qd_next)
            
        wp.launch(inplace_assign_transform, dim=self.num_bodies, inputs=[end_state.body_q], outputs=[body_q_next], device=self.device)
        wp.launch(inplace_assign_spatial_vector, dim=self.num_bodies, inputs=[end_state.body_qd], outputs=[body_qd_next], device=self.device)

        if check_diffs:
            assert requires_grad, "check_diffs requires requires_grad=True because the state gets overwritten otherwise"
            # check which arrays in model were modified
            for key, value in vars(model_before).items():
                if isinstance(value, wp.array) and len(value) > 0:
                    if not np.allclose(value.numpy(), getattr(self.model, key).numpy()):
                        print(f"model.{key} was modified")
                        print("  before:", value.numpy().flatten())
                        print("  after: ", getattr(self.model, key).numpy().flatten())
            # check which arrays in state were modified
            for key, value in vars(start_state).items():
                if isinstance(value, wp.array) and len(value) > 0:
                    if not np.allclose(value.numpy(), getattr(end_state, key).numpy()):
                        print(f"state.{key} was modified")
                        print("  before:", value.numpy().flatten())
                        print("  after: ", getattr(end_state, key).numpy().flatten())

        if (self.render):
            self._render(end_state)

    def warp_step_generalized(self, q: wp.array, qd: wp.array, tau: wp.array, q_next: wp.array, qd_next: wp.array, requires_grad=False, check_diffs=False):
        """
        Advances the system dynamics given the generalized rigid-body state and joint torques [q, qd, tau].
        Simulates for the set number of substeps and returns the next state in generalized coordinates [q_next, qd_next].
        XXX The forward kinematics and inverse kinematics projection steps may lead to known numerical issues where the
        32-bit floating point precision is not sufficient to represent the state updates, leading to a damped/frozen system.
        """

        if check_diffs:
            model_before = self.builder.finalize(self.device)

        start_state = self.model.state(requires_grad=requires_grad)

        wp.sim.eval_fk(self.model, q, qd, None, start_state)

        # assign input controls as joint torques
        wp.launch(inplace_assign, dim=self.dof_qd, inputs=[tau], outputs=[self.model.joint_act], device=self.device)
        
        end_state = self.simulate(start_state, requires_grad=requires_grad)

        wp.sim.eval_ik(self.model, end_state, q_next, qd_next)

        if check_diffs:
            assert requires_grad, "check_diffs requires requires_grad=True because the state gets overwritten otherwise"
            # check which arrays in model were modified
            for key, value in vars(model_before).items():
                if isinstance(value, wp.array) and len(value) > 0:
                    if not np.allclose(value.numpy(), getattr(self.model, key).numpy()):
                        print(f"model.{key} was modified")
                        print("  before:", value.numpy().flatten())
                        print("  after: ", getattr(self.model, key).numpy().flatten())
            # check which arrays in state were modified
            for key, value in vars(start_state).items():
                if isinstance(value, wp.array) and len(value) > 0:
                    if not np.allclose(value.numpy(), getattr(end_state, key).numpy()):
                        print(f"state.{key} was modified")
                        print("  before:", value.numpy().flatten())
                        print("  after: ", getattr(end_state, key).numpy().flatten())

        if (self.render):
            self._render(end_state)

    def generalized_deviation(self, q: np.ndarray, qd: np.ndarray):
        """
        Evaluate forward kinematics and inverse kinematics to compute the deviation between the input [q, qd]
        and the output [q', qd'] derived from the system kinematics.
        """
        wp_q = wp.array(q, device=self.device, dtype=wp.float32)
        wp_qd = wp.array(qd, device=self.device, dtype=wp.float32)
        state = self.model.state()
        wp.sim.eval_fk(self.model, wp_q, wp_qd, None, state)
        wp_q_out = wp.zeros(self.dof_q, device=self.device, dtype=wp.float32)
        wp_qd_out = wp.zeros(self.dof_qd, device=self.device, dtype=wp.float32)
        wp.sim.eval_ik(self.model, state, wp_q_out, wp_qd_out)
        # print("q_out: ", wp_q_out.numpy())
        # print("qd_out:", wp_qd_out.numpy())
        return np.concatenate([wp_q_out.numpy() - q, wp_qd_out.numpy() - qd])

    @staticmethod
    def fd_jacobian(f, x, eps=1e-5):
        num_in = len(x)
        num_out = len(f(x))
        jac = np.zeros((num_out, num_in), dtype=np.float32)
        for i in range(num_in):
            x[i] += eps
            f1 = f(x)
            x[i] -= 2*eps
            f2 = f(x)
            x[i] += eps
            jac[:, i] = (f1 - f2) / (2*eps)
        return jac

    @staticmethod
    def onehot(dim, i):
        v = np.zeros(dim, dtype=np.float32)
        v[i] = 1.0
        return v

    def fd_jacobian_maximal(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, eps=1e-5, compute_generalized_coords=True):
        # build a vector function that accepts the concatenation of (q, qd, tau) and
        # returns the concatenation of (q_next, qd_next, joint_q_next, joint_qd_next)
        def f(q_qd_tau):
            body_q = q_qd_tau[:self.num_bodies*7].reshape((self.num_bodies, 7))
            body_qd = q_qd_tau[self.num_bodies*7:self.num_bodies*(7+6)].reshape((self.num_bodies, 6))
            q = wp.array(body_q, dtype=wp.transform, device=self.device)
            qd = wp.array(body_qd, dtype=wp.spatial_vector, device=self.device)
            tau = wp.array(q_qd_tau[-self.dof_qd:], dtype=wp.float32, device=self.device)
            if compute_generalized_coords:
                joint_q_next = wp.zeros(self.dof_q, dtype=wp.float32, device=self.device)
                joint_qd_next = wp.zeros(self.dof_qd, dtype=wp.float32, device=self.device)
            else:
                joint_q_next = None
                joint_qd_next = None
            q_next = wp.zeros_like(q)
            qd_next = wp.zeros_like(qd)
            self.warp_step_maximal(q, qd, tau, q_next, qd_next, joint_q_next, joint_qd_next, requires_grad=True)
            if compute_generalized_coords:
                return np.concatenate([q_next.numpy().flatten(), qd_next.numpy().flatten(), joint_q_next.numpy(), joint_qd_next.numpy()])
            else:
                return np.concatenate([q_next.numpy().flatten(), qd_next.numpy().flatten()])

        q_qd_tau = np.concatenate([q.flatten(), qd.flatten(), tau], dtype=np.float32)
        return self.fd_jacobian(f, q_qd_tau, eps)

    def fd_jacobian_generalized(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, eps=1e-5):
        # build a vector function that accepts the concatenation of (q, qd, tau) and
        # returns the concatenation of (q_next, qd_next)
        def f(q_qd_tau):
            q = wp.array(q_qd_tau[:self.dof_q], dtype=wp.float32, device=self.device)
            qd = wp.array(q_qd_tau[self.dof_q:self.dof_q+self.dof_qd], dtype=wp.float32, device=self.device)
            tau = wp.array(q_qd_tau[-self.dof_qd:], dtype=wp.float32, device=self.device)
            q_next = wp.zeros_like(q)
            qd_next = wp.zeros_like(qd)
            self.warp_step_generalized(q, qd, tau, q_next, qd_next)
            return np.concatenate([q_next.numpy(), qd_next.numpy()])

        q_qd_tau = np.concatenate([q, qd, tau], dtype=np.float32)
        return self.fd_jacobian(f, q_qd_tau, eps)

    def ad_jacobian_maximal(self, q, qd, tau):
        q = wp.array(q, dtype=wp.transform, device=self.device, requires_grad=True)
        qd = wp.array(qd, dtype=wp.spatial_vector, device=self.device, requires_grad=True)
        tau = wp.array(tau, dtype=wp.float32, device=self.device, requires_grad=True)
        q_next = wp.zeros_like(q)
        qd_next = wp.zeros_like(qd)
        joint_q = wp.zeros(self.dof_q, dtype=wp.float32, device=self.device, requires_grad=True)
        joint_qd = wp.zeros(self.dof_qd, dtype=wp.float32, device=self.device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            self.warp_step_maximal(q, qd, tau, q_next, qd_next, joint_q, joint_qd, requires_grad=True)
        
        num_in = self.num_bodies*(7+6) + self.dof_qd
        num_out = self.num_bodies*(7+6) + self.dof_q + self.dof_qd
        jac = np.zeros((num_out, num_in), dtype=np.float32) + np.nan
        for i in range(num_out):
            # select which row of the Jacobian we want to compute
            if i < self.num_bodies*7:
                q_next.grad = wp.array(self.onehot(self.num_bodies*7, i).reshape((-1, 7)), dtype=wp.transform, device=self.device)
            elif i < self.num_bodies*(7+6):
                j = i - self.num_bodies*7
                qd_next.grad = wp.array(self.onehot(self.num_bodies*6, j).reshape((-1, 6)), dtype=wp.spatial_vector, device=self.device)
            elif i < self.num_bodies*(7+6) + self.dof_q:
                j = i - self.num_bodies*(7+6)
                joint_q.grad = wp.array(self.onehot(self.dof_q, j), dtype=wp.float32, device=self.device)
            else:
                j = i - self.num_bodies*(7+6) - self.dof_q
                joint_qd.grad = wp.array(self.onehot(self.dof_qd, j), dtype=wp.float32, device=self.device)

            tape.backward()
            jac[i, :self.num_bodies*7] = tape.gradients[q].numpy().flatten()
            jac[i, self.num_bodies*7:self.num_bodies*(7+6)] = tape.gradients[qd].numpy().flatten()
            jac[i, -self.dof_qd:] = tape.gradients[tau].numpy()
            tape.zero() 
        return jac

    def ad_jacobian_generalized(self, q, qd, tau):
        q = wp.array(q, dtype=wp.float32, device=self.device, requires_grad=True)
        qd = wp.array(qd, dtype=wp.float32, device=self.device, requires_grad=True)
        tau = wp.array(tau, dtype=wp.float32, device=self.device, requires_grad=True)
        q_next = wp.zeros_like(q)
        qd_next = wp.zeros_like(qd)

        tape = wp.Tape()
        with tape:
            self.warp_step_generalized(q, qd, tau, q_next, qd_next, requires_grad=True)
        
        num_in = self.dof_q + 2 * self.dof_qd
        num_out = self.dof_q + self.dof_qd
        jac = np.zeros((num_out, num_in), dtype=np.float32) + np.nan
        for i in range(num_out):
            # select which row of the Jacobian we want to compute
            if i < self.dof_q:
                q_next.grad = wp.array(self.onehot(self.dof_q, i), dtype=wp.float32, device=self.device)
                qd_next.grad = wp.zeros(self.dof_qd, dtype=wp.float32, device=self.device)
            else:
                q_next.grad = wp.zeros(self.dof_q, dtype=wp.float32, device=self.device)
                qd_next.grad = wp.array(self.onehot(self.dof_qd, i - self.dof_q), dtype=wp.float32, device=self.device)
            tape.backward()
            jac[i, :self.dof_q] = tape.gradients[q].numpy()
            jac[i, self.dof_q:self.dof_q+self.dof_qd] = tape.gradients[qd].numpy()
            jac[i, -self.dof_qd:] = tape.gradients[tau].numpy()
            tape.zero() 
        return jac

np.set_printoptions(precision=16, linewidth=2000, suppress=True)


sim = RigidBodySimulator(render=True, device=wp.get_preferred_device())
# sim = RigidBodySimulator(render=True, device="cpu")

use_maximal = True
if use_maximal:
    q = sim.model.body_q.numpy()
    qd = sim.model.body_qd.numpy()
else:
    q = sim.model.joint_q.numpy()
    qd = sim.model.joint_qd.numpy()
    # print(sim.generalized_deviation(q, qd))
tau = sim.model.joint_act.numpy()

# randomize inputs
np.random.seed(123)
# q = np.random.randn(*q.shape) * 0.5
# qd = np.random.randn(*qd.shape) * 6.5
tau = np.zeros(sim.dof_qd)  # np.random.randn(sim.dof_qd) * 10.0

print("q:  ", q)
print("qd: ", qd)
print("tau:", tau)

if False:
    # check backward pass
    q = wp.array(q, dtype=wp.transform, device=sim.device, requires_grad=True)
    qd = wp.array(qd, dtype=wp.spatial_vector, device=sim.device, requires_grad=True)
    tau = wp.clone(sim.model.joint_act)
    tau.requires_grad = True
    out_q = wp.zeros_like(q)
    out_qd = wp.zeros_like(qd)

    tape = wp.Tape()
    with tape:
        sim.warp_step_maximal(q, qd, tau, out_q, out_qd, requires_grad=True)
    check_backward_pass(
        tape,
        track_inputs=[q, qd, tau], track_outputs=[out_q, out_qd],
        visualize_graph=False, plot_jac_on_fail=True)

    import sys
    sys.exit(0)

if True:
    # check dynamics step Jacobian
    if use_maximal:
        jac_ad = sim.ad_jacobian_maximal(q, qd, tau)
        print("AD Jacobian:")
        print(jac_ad)

        jac_fd = sim.fd_jacobian_maximal(q, qd, tau, eps=1e-4)
        print("FD Jacobian:")
        print(jac_fd)
    else:
        jac_ad = sim.ad_jacobian_generalized(q, qd, tau)
        print("AD Jacobian:")
        print(jac_ad)

        jac_fd = sim.fd_jacobian_generalized(q, qd, tau, eps=1e-4)
        print("FD Jacobian:")
        print(jac_fd)


    print("Jacobian difference:")
    print((jac_ad - jac_fd))

    from warp.tests.grad_utils import plot_jacobian_comparison
    plot_jacobian_comparison(jac_ad, jac_fd, title="Jacobian comparison")

if True:
    # plot trajectory (joint positions)
    qs = []
    qds = []
    if use_maximal:
        q = wp.clone(sim.model.body_q)
        qd = wp.clone(sim.model.body_qd)
    else:
        q = np.zeros(sim.dof_q, dtype=np.float32)
        q[:2] = [0.0, 0.3]
        q = wp.array(q, dtype=wp.float32, device=sim.device)
        qd = wp.zeros(sim.dof_qd, dtype=wp.float32, device=sim.device)
    control = wp.zeros(sim.dof_qd, dtype=wp.float32, device=sim.device)
    for _ in trange(sim.episode_frames):
        q_next = wp.zeros_like(q)
        qd_next = wp.zeros_like(qd)
        if use_maximal:
            joint_q = wp.zeros(sim.dof_q, dtype=wp.float32, device=sim.device)
            joint_qd = wp.zeros(sim.dof_qd, dtype=wp.float32, device=sim.device)
            sim.warp_step_maximal(q, qd, control, q_next, qd_next, joint_q, joint_qd, requires_grad=True)
            qs.append(joint_q.numpy())
            qds.append(joint_qd.numpy())
        else:
            sim.warp_step_generalized(q, qd, control, q_next, qd_next, requires_grad=True)
            qs.append(q_next.numpy())
            qds.append(qd_next.numpy())
        q = q_next
        qd = qd_next

        # print(sim.generalized_deviation(q.numpy(), qd.numpy()))

    import matplotlib.pyplot as plt
    joint_q_history = np.array(qs)
    dof_q = joint_q_history.shape[1]
    ncols = int(np.ceil(np.sqrt(dof_q)))
    nrows = int(np.ceil(dof_q / float(ncols)))
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(ncols * 3.5, nrows * 3.5),
        squeeze=False,
        sharex=True
    )

    joint_id = 0
    joint_names = {
        wp.sim.JOINT_BALL.val: "ball", 
        wp.sim.JOINT_REVOLUTE.val: "hinge", 
        wp.sim.JOINT_PRISMATIC.val: "slide", 
        wp.sim.JOINT_UNIVERSAL.val: "universal",
        wp.sim.JOINT_COMPOUND.val: "compound",
        wp.sim.JOINT_FREE.val: "free", 
        wp.sim.JOINT_FIXED.val: "fixed"
    }
    joint_lower = sim.model.joint_limit_lower.numpy()
    joint_upper = sim.model.joint_limit_upper.numpy()
    joint_type = sim.model.joint_type.numpy()
    while joint_id < len(joint_type)-1 and joint_type[joint_id] == wp.sim.JOINT_FIXED.val:
        # skip fixed joints
        joint_id += 1
    q_start = sim.model.joint_q_start.numpy()
    qd_start = sim.model.joint_qd_start.numpy()
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
        ax.set_title(f"$\\mathbf{{q_{{{dim}}}}}$ ({sim.model.joint_name[joint_id]} / {joint_name} {joint_id})")
        if joint_id < sim.model.joint_count-1 and q_start[joint_id+1] == dim+1:
            joint_id += 1
            qd_i = qd_start[joint_id]
        elif qd_i < len(joint_upper)-1:
            qd_i += 1
        else:
            break
    plt.tight_layout()
    plt.show()
