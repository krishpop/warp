# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# PointMass environment
#
# Shows how to set up a simulation of a rigid-body PointMass articulation
# from a URDF using the Environment class.
# Note this example does not include a trained policy.
#
###########################################################################

import os
import math
import warp as wp
import warp.sim

from environment import Environment, run_env, IntegratorType


@wp.kernel
def cost_1d(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    target: float,
    cost: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()

    q = joint_q[env_id]
    qd = joint_qd[env_id]
    u = joint_act[env_id]

    c = (q - target) ** 2.0 + 0.1 * qd**2.0 + 0.001 * (u**2.0)

    wp.atomic_add(cost, env_id, c)


class PointMassEnvironment(Environment):
    sim_name = "env_point_mass"

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

    target = 1.0
    joint_limit = 2.0

    env_offset = (joint_limit * 2.0 + 0.2, 0.0, 2.0)

    def create_articulation(self, builder):
        box = builder.add_shape_box(
            body=-1,
            hx=self.joint_limit,
            hy=0.02,
            hz=0.02,
            density=0.0,
            has_ground_collision=False,
        )
        b = builder.add_body(name="point_mass")
        sphere = builder.add_shape_sphere(b, radius=0.15, density=1000.0, has_ground_collision=False)
        builder.add_joint_prismatic(
            parent=-1, child=b, axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=-self.joint_limit, limit_upper=self.joint_limit
        )
        builder.shape_collision_filter_pairs.add((box, sphere))

    def custom_augment_state(self, model, state):
        state.joint_q = wp.zeros(self.model.joint_q.shape, device=self.device, requires_grad=True)
        state.joint_qd = wp.zeros(self.model.joint_qd.shape, device=self.device, requires_grad=True)

    def evaluate_cost(self, state: wp.sim.State, cost: wp.array, step: int, horizon_length: int):
        wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        wp.launch(
            cost_1d,
            dim=self.num_envs,
            inputs=[state.joint_q, state.joint_qd, state.joint_act, self.target],
            outputs=[cost],
            device=self.device,
        )


if __name__ == "__main__":
    run_env(PointMassEnvironment)
