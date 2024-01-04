# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Cartpole environment with analytical dynamics formulation
###########################################################################

import os
import math
import warp as wp

import numpy as np

np.set_printoptions(precision=5, linewidth=256, suppress=True)

# wp.config.verify_cuda = True
# wp.config.mode = "debug"
# wp.verify_fp = True
import warp.sim

from environment import Environment, run_env, IntegratorType, RenderMode


if False:

    @wp.kernel
    def single_cartpole_cost(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        cost: wp.array(dtype=wp.float32),
    ):
        env_id = wp.tid()

        pos_cart = wp.transform_get_translation(body_q[env_id * 2])
        # cart must be at target (x = 3)
        cart_cost = (3.0 - pos_cart[0]) ** 2.0

        vel_cart = body_qd[env_id * 2]

        # encourage zero velocity
        vel_cost = wp.length_sq(vel_cart)

        # wp.atomic_add(cost, env_id, (cart_cost + 0.02 * vel_cost))
        wp.atomic_add(cost, env_id, cart_cost + 0.02 * vel_cost)

else:

    @wp.kernel
    def single_cartpole_cost2(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        cost: wp.array(dtype=wp.float32),
    ):
        env_id = wp.tid()

        pos_cart = wp.transform_get_translation(body_q[env_id * 2])
        pos_pole = wp.transform_vector(body_q[env_id * 2 + 1], wp.vec3(0.0, 0.0, 1.0))
        # wp.printf("[%.3f %.3f %.3f]\n", pos_pole[0], pos_pole[1], pos_pole[2])

        # cart must be at the origin (x = 0)
        cart_cost = pos_cart[0] ** 2.0
        # pole must be upright (x = 0, y as high as possible)
        # pole_cost = pos_pole[0] ** 2.0 - 0.1 * pos_pole[1]
        pole_cost = (1.0 - pos_pole[1]) ** 2.0 * 1000.0 + (pos_cart[0] - pos_pole[0]) ** 2.0 * 10.0

        # wp.printf("pos_pole = [%.3f %.3f %.3f]\n", pos_pole[0], pos_pole[1], pos_pole[2])

        vel_cart = body_qd[env_id * 2]
        vel_pole = body_qd[env_id * 2 + 1]

        # encourage zero velocity
        vel_cost = wp.length_sq(vel_cart) + wp.length_sq(vel_pole)

        wp.atomic_add(cost, env_id, 1.0e-2 * pole_cost)  # (10.0 * (cart_cost + 0.0 * pole_cost) + 0.0 * vel_cost))
        # wp.atomic_add(cost, env_id, 10.0 * (cart_cost + pole_cost) + 0.02 * vel_cost)

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

        th = joint_q[env_id * 2 + 1]
        thdot = joint_qd[env_id * 2 + 1]
        u = joint_act[env_id * 2]

        # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
        c = angle_normalize(th) ** 2.0 + 0.1 * thdot**2.0 + (u * 1e-4) ** 2.0

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
    # from https://courses.ece.ucsb.edu/ECE594/594D_W10Byl/hw/cartpole_eom.pdf
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

    mlsinth = mass_pole * pole_length * sinth
    fterm = f + mlsinth * thd * thd

    denom = pole_length * (costh * costh * mass_pole - msum)
    xdd = (-gravity * mlsinth * costh - pole_length * fterm) / denom
    thdd = (msum * gravity * sinth + costh * fterm) / denom

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
    # env_offset = (0.0, 0.0, 0.0)

    single_cartpole = False

    opengl_render_settings = dict(scaling=3.0, draw_axis=False)
    usd_render_settings = dict(scaling=100.0)

    activate_ground_plane = False
    show_joints = False

    controllable_dofs = [0]
    control_gains = [500.0]
    control_limits = [(-1.0, 1.0)]

    integrator_type = IntegratorType.FEATHERSTONE
    num_envs = 100

    requires_grad = False
    use_graph_capture = True

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
        builder.joint_q[1] = 0.1

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
