# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Sticky contact environment
###########################################################################

import warp as wp
import warp.sim

import numpy as np

from environment import Environment, run_env, IntegratorType


@wp.kernel
def apply_forces(
    time: float,
    dof_per_env: int,
    joint_act: wp.array(dtype=float),
    shape_materials: wp.sim.ModelShapeMaterials,
):
    tid = wp.tid()
    dof = 6
    if time < 0.5:
        joint_act[tid * dof_per_env + dof] = 110.0
    else:
        joint_act[tid * dof_per_env + dof] = 1000.0
    if time > 2.5:
        # remove adhesion
        shape_materials.ka[tid * 2 + 0] = 0.0
        shape_materials.ka[tid * 2 + 1] = 0.0


class StickyContactEnvironment(Environment):
    sim_name = "env_sticky_contact"
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
    contact_points_radius = 0.02

    episode_duration = 5.0

    integrator_type = IntegratorType.FEATHERSTONE

    def create_articulation(self, builder):
        b1 = builder.add_body()
        builder.add_shape_box(b1, hx=0.2, hy=0.2, hz=0.2)
        builder.add_joint_free(child=b1, parent_xform=wp.transform(wp.vec3(0.0, 0.2, 0.0)))

        b2 = builder.add_body()
        builder.add_shape_box(b2, hx=0.15, hy=0.15, hz=0.15, ka=0.1, thickness=0.01)
        builder.add_joint_prismatic(
            axis=wp.vec3(0.0, 1.0, 0.0), parent=-1, child=b2, parent_xform=wp.transform(wp.vec3(0.0, 0.5 + 0.15, 0.0))
        )
        builder.joint_q[-1] = 0.8

        self.dof_per_env = len(builder.joint_qd)

    def custom_update(self):
        # print(self.state.joint_q.numpy(), self.state.joint_act.numpy())
        # print(self.model.shape_materials.ka.numpy())
        wp.launch(
            apply_forces,
            dim=self.num_envs,
            inputs=[self.sim_time, self.dof_per_env],
            outputs=[self.state.joint_act, self.model.shape_materials],
        )


if __name__ == "__main__":
    run_env(StickyContactEnvironment)
