# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Cartpole environment with analytical dynamics formulation
###########################################################################

import warp as wp

import numpy as np

np.set_printoptions(precision=5, linewidth=256, suppress=True)

# wp.config.verify_cuda = True
# wp.config.mode = "debug"
# wp.verify_fp = True
import warp.sim

from environment import Environment, run_env, IntegratorType


@wp.func
def angle_normalize(x: float):
    return ((x + wp.pi) % (2.0 * wp.pi)) - wp.pi


@wp.kernel
def single_cartpole_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    cost: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()

    x = joint_q[env_id * 2 + 0]
    th = joint_q[env_id * 2 + 1]
    thdot = joint_qd[env_id * 2 + 1]
    u = joint_act[env_id * 2]

    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
    c = angle_normalize(th) ** 2.0 + 0.1 * thdot**2.0 + (u * 1e-4) ** 2.0 + 1.0 * x**2.0

    wp.atomic_add(cost, env_id, c)


@wp.kernel
def cartpole_dynamics(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    params: wp.array(dtype=wp.float32, ndim=2),
    gravity: float,
    dt: float,
    # outputs
    joint_q_next: wp.array(dtype=wp.float32),
    joint_qd_next: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    # Eq. 24, 25 from https://courses.ece.ucsb.edu/ECE594/594D_W10Byl/hw/cartpole_eom.pdf
    f = joint_act[env_id * 2]
    x = joint_q[env_id * 2 + 0]
    th = joint_q[env_id * 2 + 1]
    xd = joint_qd[env_id * 2 + 0]
    thd = joint_qd[env_id * 2 + 1]

    mass_cart = params[env_id, 0]
    mass_pole = params[env_id, 1]
    pole_length = params[env_id, 2]

    sinth = wp.sin(th)
    costh = wp.cos(th)
    msum = mass_cart + mass_pole
    denom = mass_cart + mass_pole * (1.0 - costh * costh)
    mpstt = -mass_pole * pole_length * sinth * thd * thd
    gs = -gravity * sinth

    thdd = mpstt * costh + gs * msum + costh * f
    thdd /= pole_length * denom

    xdd = mpstt + mass_pole * gs * costh + f
    xdd /= denom

    # semi-implicit Euler
    xd_next = xd + dt * xdd
    x_next = x + dt * xd_next
    thd_next = thd + dt * thdd
    th_next = th + dt * thd_next

    joint_q_next[env_id * 2 + 0] = x_next
    joint_q_next[env_id * 2 + 1] = th_next
    joint_qd_next[env_id * 2 + 0] = xd_next
    joint_qd_next[env_id * 2 + 1] = thd_next


class AnalyticalCartpoleEnvironment(Environment):
    sim_name = "env_analytical_cartpole"
    env_offset = (2.0, 0.0, 2.0)

    start_angle = 0.1  # radians

    opengl_render_settings = dict(scaling=3.0, draw_axis=False)
    usd_render_settings = dict(scaling=100.0)

    activate_ground_plane = False
    show_joints = False

    controllable_dofs = [0]
    control_gains = [500.0]
    control_limits = [(-1.0, 1.0)]

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_euler = 1
    num_envs = 100

    default_mass_cart = 0.5
    default_mass_pole = 0.3
    default_pole_length = 0.4

    def create_articulation(self, builder):
        builder.add_shape_box(hx=10.0, hy=0.01, hz=0.01, density=0, body=-1)

        cart = builder.add_body(name="cart")
        builder.add_shape_box(hx=0.05, hy=0.03, hz=0.03, density=0, body=cart)

        builder.add_joint_prismatic(parent=-1, child=cart, axis=wp.vec3(1.0, 0.0, 0.0))

        pole = builder.add_body(name="pole")
        builder.add_shape_cylinder(
            radius=0.02,
            up_axis=1,
            half_height=self.default_pole_length / 2,
            pos=(0.0, self.default_pole_length / 2, 0.0),
            body=pole,
        )
        builder.add_shape_sphere(radius=0.04, pos=(0.0, self.default_pole_length, 0.0), body=pole)

        builder.add_joint_revolute(parent=cart, child=pole, axis=wp.vec3(0.0, 0.0, 1.0))

        # pole angle
        builder.joint_q[1] = self.start_angle

    def customize_model(self, model):
        # allocate model parameters
        params = np.zeros((self.num_envs, 3), dtype=np.float32)
        params[:, 0] = self.default_mass_cart
        params[:, 1] = self.default_mass_pole
        params[:, 2] = self.default_pole_length
        model.params = wp.array(params, device=model.device, requires_grad=model.requires_grad)

    def custom_dynamics(self, model, state, next_state, dt):
        wp.launch(
            cartpole_dynamics,
            dim=self.num_envs,
            inputs=[state.joint_q, state.joint_qd, state.joint_act, model.params, model.gravity[1], dt],
            outputs=[next_state.joint_q, next_state.joint_qd],
        )

    def custom_render(self, render_state):
        # update body transforms for rendering
        wp.sim.eval_fk(self.model, render_state.joint_q, render_state.joint_qd, None, render_state)

    def evaluate_cost(self, state: wp.sim.State, cost: wp.array, step: int, horizon_length: int):
        wp.launch(
            single_cartpole_cost,
            dim=self.num_envs,
            inputs=[state.joint_q, state.joint_qd, state.joint_act],
            outputs=[cost],
            device=self.device,
        )


if __name__ == "__main__":
    run_env(AnalyticalCartpoleEnvironment)
