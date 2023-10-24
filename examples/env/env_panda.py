# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Panda environment
#
# Shows how to set up a simulation of a rigid-body Hopper articulation based on
# the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import warp as wp
import warp.sim

import numpy as np

from environment import Environment, run_env


@wp.kernel
def panda_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    cost: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    base_tf = body_q[env_id * 4 + 2]

    pos_base = wp.transform_get_translation(base_tf)
    rot_base = wp.transform_get_rotation(base_tf)
    vel_base = body_qd[env_id * 4 + 2]

    # cost[env_id] = cost[env_id] + (vel_base[4]) + pos_base[0] * 10.0
    # cost[env_id] = 0.95 * cost[env_id] + 10.0 * (cart_cost + pole_cost) + 0.02 * vel_cost

    termination_height = 0.17

    up_vec = wp.quat_rotate(rot_base, wp.vec3(0.0, 1.0, 0.0))
    heading_vec = wp.quat_rotate(rot_base, wp.vec3(1.0, 0.0, 0.0))

    # wp.printf("up_vec: [%.3f %.3f %.3f]\n", up_vec[0], up_vec[1], up_vec[2])
    # wp.printf("heading_vec: [%.3f %.3f %.3f]\n", heading_vec[0], heading_vec[1], heading_vec[2])

    up_reward = wp.length_sq(up_vec - wp.vec3(0.0, 0.0, -1.0))
    heading_reward = wp.length_sq(heading_vec - wp.vec3(1.0, 0.0, 0.0))
    height_reward = pos_base[1] - termination_height
    progress_reward = vel_base[3]  # double-check!

    reward = progress_reward  # + 0.001 * (up_reward + heading_reward + height_reward)
    cost[env_id] = cost[env_id] - reward


@wp.kernel
def apply_forces(time: float, act_dim: int, joint_act: wp.array(dtype=float)):
    tid = wp.tid()
    joint_act[tid * act_dim + 1] = -12.0  # XXX need to revert force direction because joint is rotated
    # joint_act[tid*act_dim + 2] = wp.sin(time * 0.5)
    joint_act[tid * act_dim + 3] = wp.sin(time * 0.5)


class PandaEnvironment(Environment):
    sim_name = "env_panda"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    xpbd_settings = dict(iterations=5)

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    use_graph_capture = False
    use_tiled_rendering = False
    show_joints = False

    show_rigid_contact_points = True

    controllable_dofs = [3, 4, 5]
    control_gains = [100.0] * 3
    control_limits = [(-1.0, 1.0)] * 3
    
    # requires_grad = True
    episode_duration = 1.1

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            r"C:\Users\eric-\source\repos\warp-tool-use-yewon\examples\assets\panda\panda_gripper.urdf",
            # r"C:\Users\eheiden\Documents\warp-tool-use-yewon\examples\assets\panda\panda_gripper_nostick.urdf",
            # r"F:\Projects\warp-tool-use-yewon\svgd\assets\panda\panda_gripper.urdf",
            builder,
            # xform=wp.transform(np.array([(i//10)*1.0, 0.30, (i%10)*1.0]), wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi*0.5)),
            # xform=wp.transform(wp.array(panda_pos), wp.quat_from_axis_angle((1.0, 0.0, 0.0), 0)),
            xform=wp.transform(
                np.array([0., 0.30, 0.]),
                wp.quat_identity(),
                # wp.quat_from_axis_angle((0.0, 0.0, 1.0), -np.pi * 0.5)
            ),
            floating=False,
            base_joint="px, py, pz, ry",
            density=1000,
            armature=0.001,
            stiffness=0.0,
            damping=10,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=1.5,
            # shape_thickness=self.contact_thickness,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            parse_visuals_as_colliders=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True)
        self.act_dim = len(builder.joint_act)
        print("act dof:", self.act_dim)

    def evaluate_cost(self, state: wp.sim.State, cost: wp.array, step: int, horizon_length: int):
        wp.launch(
            panda_cost,
            dim=self.num_envs,
            inputs=[state.body_q, state.body_qd],
            outputs=[cost],
            device=self.device
        )

    def custom_update(self):
        wp.launch(apply_forces,
                  dim=self.num_envs,
                  inputs=[self.sim_time, self.act_dim],
                  outputs=[self.state.joint_act],)


if __name__ == "__main__":
    run_env(PandaEnvironment)
