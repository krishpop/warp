# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Cartpole environment
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a URDF using the Environment class.
# Note this example does not include a trained policy.
#
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


# @wp.kernel
# def single_cartpole_cost(
#     body_q: wp.array(dtype=wp.transform),
#     body_qd: wp.array(dtype=wp.spatial_vector),
#     cost: wp.array(dtype=wp.float32),
# ):
#     env_id = wp.tid()

#     pos_cart = wp.transform_get_translation(body_q[env_id])
#     # cart must be at target (x = 3)
#     cart_cost = (3.0 - pos_cart[0]) ** 2.0

#     vel_cart = body_qd[env_id]

#     # encourage zero velocity
#     vel_cost = wp.length_sq(vel_cart)

#     # wp.atomic_add(cost, env_id, (cart_cost + 0.02 * vel_cost))
#     wp.atomic_add(cost, env_id, cart_cost + 0.02 * vel_cost)

if False:
    @wp.kernel
    def single_cartpole_cost(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        cost: wp.array(dtype=wp.float32),
    ):
        env_id = wp.tid()

        pos_cart = wp.transform_get_translation(body_q[env_id*2])
        # cart must be at target (x = 3)
        cart_cost = (3.0 - pos_cart[0]) ** 2.0

        vel_cart = body_qd[env_id*2]

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

        wp.atomic_add(cost, env_id, 1.0e-2 * pole_cost) # (10.0 * (cart_cost + 0.0 * pole_cost) + 0.0 * vel_cost))
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
        c = angle_normalize(th) ** 2.0 + 0.1 * thdot ** 2.0 + (u * 1e-4) ** 2.0

        wp.atomic_add(cost, env_id, c)


@wp.kernel
def double_cartpole_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    cost: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()

    pos_cart = wp.transform_get_translation(body_q[env_id * 3])
    pos_pole_1 = wp.transform_vector(body_q[env_id * 3 + 1], wp.vec3(0.0, 0.0, 1.0))
    pos_pole_2 = wp.transform_vector(body_q[env_id * 3 + 2], wp.vec3(0.0, 0.0, 1.0))

    # cart must be at the origin (z = 0)
    cart_cost = pos_cart[2] ** 2.0
    # pole must be upright (z = 0, y as high as possible)
    pole_cost = pos_pole_1[2] ** 2.0 - pos_pole_1[1]
    pole_cost += pos_pole_2[2] ** 2.0 - pos_pole_2[1]

    vel_cart = body_qd[env_id * 3]
    vel_pole = body_qd[env_id * 3 + 1]

    wp.printf("pos_cart = [%.3f %.3f %.3f]\n", pos_cart[0], pos_cart[1], pos_cart[2])

    # encourage zero velocity
    vel_cost = wp.length_sq(vel_cart) + wp.length_sq(vel_pole)

    wp.atomic_add(cost, env_id, 10.0 * (cart_cost + pole_cost) + vel_cost)


class CartpoleEnvironment(Environment):
    sim_name = "env_cartpole"
    env_offset = (2.0, 0.0, 2.0)
    # env_offset = (0.0, 0.0, 0.0)

    single_cartpole = False

    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    # xpbd_settings = dict(iterations=3, joint_linear_relaxation=0.3, joint_angular_relaxation=0.1)
    # xpbd_settings = dict(iterations=0)
    # gravity = 0.0

    activate_ground_plane = False
    show_joints = True

    controllable_dofs = [0]
    control_gains = [500.0]
    control_limits = [(-1.0, 1.0)]

    integrator_type = IntegratorType.FEATHERSTONE
    num_envs = 3

    requires_grad = False
    use_graph_capture = False

    # render_mode = RenderMode.NONE
    # use_graph_capture = False
    # num_envs = 1

    plot_joint_coords = False

    eval_fk = False

    def create_articulation(self, builder):
        if self.single_cartpole:
            path = "../assets/cartpole_single.urdf"
            # path = "../assets/cartpole_cart_only.urdf"
        else:
            path = "../assets/cartpole.urdf"
            self.opengl_render_settings["camera_pos"] = (40.0, 1.0, 0.0)
            self.opengl_render_settings["camera_front"] = (-1.0, 0.0, 0.0)
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), path),
            builder,
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=False,
            density=1000,
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e4,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=0.0,
            enable_self_collisions=False,
        )
        # builder.collapse_fixed_joints()

        # joint initial positions
        builder.joint_q[-3:] = [0.0, 0.1, 0.0]

    def custom_augment_state(self, model, state):
        if self.integrator_type != IntegratorType.FEATHERSTONE:
            state.joint_q = wp.zeros(self.model.joint_q.shape, device=self.device, requires_grad=True)
            state.joint_qd = wp.zeros(self.model.joint_qd.shape, device=self.device, requires_grad=True)

    def evaluate_cost(self, state: wp.sim.State, cost: wp.array, step: int, horizon_length: int):
        
        # wp.launch(
        #     single_cartpole_cost2 if self.single_cartpole else double_cartpole_cost,
        #     dim=self.num_envs,
        #     inputs=[state.body_q, state.body_qd],
        #     outputs=[cost],
        #     device=self.device
        # )

        if self.integrator_type != IntegratorType.FEATHERSTONE:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        wp.launch(
            single_cartpole_cost if self.single_cartpole else double_cartpole_cost,
            dim=self.num_envs,
            inputs=[state.joint_q, state.joint_qd, state.joint_act],
            outputs=[cost],
            device=self.device
        )


if __name__ == "__main__":
    run_env(CartpoleEnvironment)
