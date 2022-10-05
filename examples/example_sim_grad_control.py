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


import os
import math

import numpy as np

import warp as wp
# wp.config.mode = "debug"
import warp.sim
import warp.sim.render


# wp.config.verify_fp = True
wp.init()

@wp.kernel
def inplace_assign(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    b[tid] = a[tid]

class RigidBodySimulator:
    """
    Differentiable simulator of a rigid-body system with contacts.
    The system state is described entirely by the joint positions q, velocities qd, and
    joint torques tau. The system state is updated by calling the warp_step() function.
    """

    frame_dt = 1.0/60.0

    episode_duration = 20.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 1
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
                wp.sim.parse_urdf(os.path.join(os.path.dirname(__file__), "assets/cartpole_single.urdf"), builder,
                    xform=wp.transform(np.array((0.0, 0.0, 0.0)), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
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
                # joint initial positions
                builder.joint_q[-2:] = [0.0, 0.3]
                builder.joint_target[:2] = [0.0, 0.0]
            else:
                wp.sim.parse_urdf(os.path.join(os.path.dirname(__file__), "assets/cartpole.urdf"), builder,
                    xform=wp.transform(np.array((i*2.0, 4.0, 0.0)), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
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

        # finalize model
        self.model = builder.finalize(device)
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

        self.state = self.model.state()

        self.solve_iterations = 10
        # self.integrator = wp.sim.XPBDIntegrator(self.solve_iterations)
        self.integrator = wp.sim.SemiImplicitIntegrator()
        
        if (self.model.ground):
            self.model.collide(self.state)

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_cartpole.usd"))

    def warp_step(self, q, qd, tau, q_next, qd_next, requires_grad=False):
        if requires_grad:
            # ground = self.model.ground
            # self.model = self.builder.finalize(self.device)
            # self.model.ground = ground

            self.model.joint_act.requires_grad = True
            self.model.body_q.requires_grad = True
            self.model.body_qd.requires_grad = True

            self.model.body_mass.requires_grad = True
            self.model.body_inv_mass.requires_grad = True
            self.model.body_inertia.requires_grad = True
            self.model.body_inv_inertia.requires_grad = True
            self.model.body_com.requires_grad = True

            # just enable requires_grad for all arrays in the model
            # for name in dir(self.model):
            #     attr = getattr(self.model, name)
            #     if isinstance(attr, wp.array):
            #         attr.requires_grad = True
            
            # XXX activate requires_grad for all arrays in the material struct
            self.model.shape_materials.ke.requires_grad = True
            self.model.shape_materials.kd.requires_grad = True
            self.model.shape_materials.kf.requires_grad = True
            self.model.shape_materials.mu.requires_grad = True
            self.model.shape_materials.restitution.requires_grad = True
            
            states = [self.model.state(requires_grad=True) for _ in range(self.sim_substeps+1)]
        else:
            # states = [self.state for _ in range(self.sim_substeps+1)]
            states = [self.model.state(requires_grad=False) for _ in range(self.sim_substeps+1)]

        wp.sim.eval_fk(self.model, q, qd, None, states[0])

        # assign input controls as joint torques
        wp.launch(inplace_assign, dim=self.dof_qd, inputs=[tau], outputs=[self.model.joint_act], device=self.device)
        
        for i in range(self.sim_substeps):
            states[i].clear_forces()
            if self.model.ground:
                self.model.allocate_rigid_contacts()
                wp.sim.collide(self.model, states[i])
            self.integrator.simulate(self.model, states[i], states[i+1], self.sim_dt)

        wp.sim.eval_ik(self.model, states[-1], q_next, qd_next)    

    def fd_jacobian(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, eps=1e-5):
        # build a vector function that accepts the concatenation of (q, qd, tau) and
        # returns the concatenation of (q_next, qd_next)
        def f(q_qd_tau):
            q = wp.array(q_qd_tau[:self.dof_q], dtype=wp.float32, device=self.device)
            qd = wp.array(q_qd_tau[self.dof_q:self.dof_q+self.dof_qd], dtype=wp.float32, device=self.device)
            tau = wp.array(q_qd_tau[-self.dof_qd:], dtype=wp.float32, device=self.device)
            q_next = wp.zeros_like(q)
            qd_next = wp.zeros_like(qd)
            self.warp_step(q, qd, tau, q_next, qd_next)
            return np.concatenate([q_next.numpy(), qd_next.numpy()])

        q_qd_tau = np.concatenate([q, qd, tau], dtype=np.float32)
        num_in = self.dof_q + 2 * self.dof_qd
        num_out = self.dof_q + self.dof_qd
        jac = np.zeros((num_out, num_in), dtype=np.float32) + np.nan
        for i in range(len(q_qd_tau)):
            q_qd_tau[i] += eps
            f1 = f(q_qd_tau)
            q_qd_tau[i] -= 2*eps
            f2 = f(q_qd_tau)
            q_qd_tau[i] += eps
            jac[:, i] = (f1 - f2) / (2*eps)
        return jac

    def ad_jacobian(self, q, qd, tau):
        q = wp.array(q, dtype=wp.float32, device=self.device, requires_grad=True)
        qd = wp.array(qd, dtype=wp.float32, device=self.device, requires_grad=True)
        tau = wp.array(tau, dtype=wp.float32, device=self.device, requires_grad=True)
        q_next = wp.zeros_like(q)
        qd_next = wp.zeros_like(qd)
        q_next.requires_grad = True
        qd_next.requires_grad = True

        tape = wp.Tape()
        with tape:
            self.warp_step(q, qd, tau, q_next, qd_next, requires_grad=True)

        def onehot(dim, i):
            v = np.zeros(dim, dtype=np.float32)
            v[i] = 1.0
            return v
        
        num_in = self.dof_q + 2 * self.dof_qd
        num_out = self.dof_q + self.dof_qd
        jac = np.zeros((num_out, num_in), dtype=np.float32) + np.nan
        for i in range(num_out):
            # select which row of the Jacobian we want to compute
            if i < self.dof_q:
                q_next.grad = wp.array(onehot(self.dof_q, i), dtype=wp.float32, device=self.device)
                qd_next.grad = wp.zeros(self.dof_qd, dtype=wp.float32, device=self.device)
            else:
                q_next.grad = wp.zeros(self.dof_q, dtype=wp.float32, device=self.device)
                qd_next.grad = wp.array(onehot(self.dof_qd, i - self.dof_q), dtype=wp.float32, device=self.device)
            tape.backward()
            jac[i, :self.dof_q] = tape.gradients[q].numpy()
            jac[i, self.dof_q:self.dof_q+self.dof_qd] = tape.gradients[qd].numpy()
            jac[i, -self.dof_qd:] = tape.gradients[tau].numpy()
            tape.zero() 
        return jac

np.set_printoptions(precision=10, linewidth=200, suppress=True)

sim = RigidBodySimulator()

q = sim.model.joint_q.numpy()
qd = sim.model.joint_qd.numpy()
tau = sim.model.joint_act.numpy()

# randomize inputs
np.random.seed(123)
q = np.random.randn(sim.dof_q) * 4.5
qd = np.random.randn(sim.dof_q) * 6.5
tau = np.random.randn(sim.dof_qd) * 10.0

print("q:  ", q)
print("qd: ", qd)
print("tau:", tau)

jac_ad = sim.ad_jacobian(q, qd, tau)
print("AD Jacobian:")
print(jac_ad)

jac_fd = sim.fd_jacobian(q, qd, tau, eps=1e-4)
print("FD Jacobian:")
print(jac_fd)
