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
import warp.sim

from environment import Environment, run_env


@wp.kernel
def single_cartpole_cost(
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

    vel_cart = body_qd[env_id * 2]
    vel_pole = body_qd[env_id * 2 + 1]

    # encourage zero velocity
    vel_cost = wp.length_sq(vel_cart) + wp.length_sq(vel_pole)

    cost[env_id] = cost[env_id] + 10.0 * (cart_cost + pole_cost) + 0.02 * vel_cost


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

    # encourage zero velocity
    vel_cost = wp.length_sq(vel_cart) + wp.length_sq(vel_pole)

    cost[env_id] = cost[env_id] + 10.0 * (cart_cost + pole_cost) + vel_cost


class CartpoleEnvironment(Environment):
    sim_name = "env_cartpole"
    env_offset = (2.0, 0.0, 2.0)

    single_cartpole = False

    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    activate_ground_plane = False
    show_joints = True

    controllable_dofs = [0]
    control_gains = [500.0]
    control_limits = [(-1.0, 1.0)]

    def create_articulation(self, builder):
        if self.single_cartpole:
            path = "../assets/cartpole_single.urdf"
        else:
            path = "../assets/cartpole.urdf"
            self.opengl_render_settings["camera_pos"] = (40.0, 1.0, 0.0)
            self.opengl_render_settings["camera_front"] = (-1.0, 0.0, 0.0)
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), path),
            builder,
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=False,
            density=1000.0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e4,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
        )
        builder.collapse_fixed_joints()

        # joint initial positions
        builder.joint_q[-3:] = [0.0, 0.1, 0.0]

    def evaluate_cost(self, state: wp.sim.State, cost: wp.array):
        wp.launch(
            single_cartpole_cost if self.single_cartpole else double_cartpole_cost,
            dim=self.num_envs,
            inputs=[state.body_q, state.body_qd],
            outputs=[cost],
            device=self.device
        )


if __name__ == "__main__":
    run_env(CartpoleEnvironment)
