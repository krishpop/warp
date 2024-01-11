# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Pendulum environment
#
# Shows how to set up a simulation of a rigid-body Hopper articulation based on
# the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import numpy as np

import warp as wp
import warp.sim

from environment import Environment, run_env, IntegratorType


class PendulumEnvironment(Environment):
    sim_name = "env_pendulum"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 16
    sim_substeps_xpbd = 5

    integrator_type = IntegratorType.FEATHERSTONE

    xpbd_settings = dict(iterations=7)

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    use_graph_capture = False
    use_tiled_rendering = False
    show_joints = True

    activate_ground_plane = False
    
    num_envs = 1

    def create_articulation(self, builder):
        # self.joint_type = wp.sim.JOINT_BALL
        self.joint_type = wp.sim.JOINT_REVOLUTE
        self.chain_length = 3
        self.chain_width = 1.0

        self.lower = -np.deg2rad(60.0)
        self.upper = np.deg2rad(60.0)
        self.limitd_ke = 1e5
        self.limitd_kd = 1e2

        self.target = np.deg2rad(15)
        self.target_ke = 0.0  # 1e4
        self.target_kd = 0.0

        shape_ke = 1.0e4
        shape_kd = 1.0e2
        shape_kf = 1.0e4

        builder.set_ground_plane(
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
        )

        for i in range(self.chain_length):
            if i == 0:
                parent = -1
                parent_joint_xform = wp.transform([0.0, 2.0, 1.0], wp.quat_identity())
            else:
                parent = builder.joint_count - 1
                parent_joint_xform = wp.transform([self.chain_width, 0.0, 0.0], wp.quat_identity())

            # create body
            b = builder.add_body(origin=wp.transform([i, 0.0, 1.0], wp.quat_identity()), armature=0.1)

            # create shape
            builder.add_shape_box(
                pos=(self.chain_width * 0.5, 0.0, 0.0),
                hx=self.chain_width * 0.5,
                hy=0.1,
                hz=0.1,
                density=10.0,
                body=b,
            )

            if self.joint_type == wp.sim.JOINT_REVOLUTE:
                if i == 0:
                    builder.add_joint_revolute(
                        parent=parent,
                        child=b,
                        axis=(0.0, 0.0, 1.0),
                        parent_xform=parent_joint_xform,
                        limit_lower=self.lower,
                        limit_upper=self.upper,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                    )
                else:
                    builder.add_joint_revolute(
                        parent=parent,
                        child=b,
                        axis=(0.0, 0.0, 1.0),
                        parent_xform=parent_joint_xform,
                        # child_xform=parent_joint_xform,
                        limit_lower=self.lower,
                        limit_upper=self.upper,
                        target=self.target,
                        target_ke=self.target_ke,
                        target_kd=self.target_kd,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                    )

            elif self.joint_type == wp.sim.JOINT_UNIVERSAL:
                builder.add_joint_universal(
                    parent=parent,
                    child=b,
                    axis_0=wp.sim.JointAxis(
                        (1.0, 0.0, 0.0), self.lower, self.upper, limit_ke=self.limitd_ke, limit_kd=self.limitd_kd
                    ),
                    axis_1=wp.sim.JointAxis(
                        (0.0, 0.0, 1.0),
                        self.lower,
                        self.upper,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                        target=self.target,
                        target_ke=self.target_ke,
                        target_kd=self.target_kd,
                    ),
                    parent_xform=parent_joint_xform,
                )

            elif self.joint_type == wp.sim.JOINT_BALL:
                builder.add_joint_ball(
                    parent=parent,
                    child=b,
                    parent_xform=parent_joint_xform,
                    armature=10.0,
                )

            elif self.joint_type == wp.sim.JOINT_FIXED:
                builder.add_joint_fixed(
                    parent=parent,
                    child=b,
                    parent_xform=parent_joint_xform,
                )

            elif self.joint_type == wp.sim.JOINT_COMPOUND:
                builder.add_joint_compound(
                    parent=parent,
                    child=b,
                    axis_0=wp.sim.JointAxis(
                        (1.0, 0.0, 0.0), self.lower, self.upper, limit_ke=self.limitd_ke, limit_kd=self.limitd_kd
                    ),
                    axis_1=wp.sim.JointAxis(
                        (0.0, 1.0, 0.0), self.lower, self.upper, limit_ke=self.limitd_ke, limit_kd=self.limitd_kd
                    ),
                    axis_2=wp.sim.JointAxis(
                        (0.0, 0.0, 1.0),
                        self.lower,
                        self.upper,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                        target=self.target,
                        target_ke=self.target_ke,
                        target_kd=self.target_kd,
                    ),
                    parent_xform=parent_joint_xform,
                )

            elif self.joint_type == wp.sim.JOINT_D6:
                builder.add_joint_d6(
                    parent=parent,
                    child=b,
                    angular_axes=[
                        wp.sim.JointAxis(
                            (1.0, 0.0, 0.0), self.lower, self.upper, limit_ke=self.limitd_ke, limit_kd=self.limitd_kd
                        ),
                        wp.sim.JointAxis(
                            (0.0, 1.0, 0.0), self.lower, self.upper, limit_ke=self.limitd_ke, limit_kd=self.limitd_kd
                        ),
                        wp.sim.JointAxis(
                            (0.0, 0.0, 1.0),
                            self.lower,
                            self.upper,
                            limit_ke=self.limitd_ke,
                            limit_kd=self.limitd_kd,
                            target=self.target,
                            target_ke=self.target_ke,
                            target_kd=self.target_kd,
                        ),
                    ],
                    parent_xform=parent_joint_xform,
                )

    def custom_update(self):
        wp.launch(apply_forces,
                  dim=self.num_envs,
                  inputs=[self.sim_time, self.act_dim],
                  outputs=[self.state.joint_act],)
        builder.joint_act[0] = 100.0


if __name__ == "__main__":
    run_env(PendulumEnvironment)
