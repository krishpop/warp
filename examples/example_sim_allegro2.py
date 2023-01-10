# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Allegro
#
# Shows how to set up a simulation of a rigid-body Allegro hand articulation 
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import os

import numpy as np
import warp as wp

import warp as wp
import warp.sim

from sim_demo import WarpSimDemonstration, run_demo

class Demo(WarpSimDemonstration):
    sim_name = "example_sim_allegro"
    env_offset=(6.0, 0.0, 6.0)
    tiny_render_settings = dict(scaling=15.0)
    usd_render_settings = dict(scaling=200.0)

    sim_substeps_euler = 128
    sim_substeps_xpbd = 8

    xpbd_settings = dict(
        iterations=20,
        joint_positional_relaxation=1.0,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )
    
    def create_articulation(self, builder):
        floating_base = False
        wp.sim.parse_urdf(
            os.path.join(
                os.path.dirname(__file__),
                "assets/isaacgymenvs/kuka_allegro_description/allegro.urdf"),
            builder,
            xform=wp.transform(np.array((0.0, 0.3, 0.0)), wp.quat_rpy(-np.pi/2, np.pi*0.75, np.pi/2)),
            floating=floating_base,
            density=1e3,
            armature=0.0,
            stiffness=1000.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False)
        
        # ensure all joint positions are within limits
        q_offset = (7 if floating_base else 0)
        qd_offset = (6 if floating_base else 0)
        for i in range(16):
            builder.joint_q[i+q_offset] = 0.5 * (builder.joint_limit_lower[i+qd_offset] + builder.joint_limit_upper[i+qd_offset])
            builder.joint_target[i] = builder.joint_q[i+q_offset]
            builder.joint_target_ke[i] = 50000000.0
            builder.joint_target_kd[i] = 10.0
        
        wp.sim.parse_urdf(
            os.path.join(
                os.path.dirname(__file__),
                "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf"),
            builder,
            xform=wp.transform(np.array((-0.1, 0.5, 0.0)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            parse_visuals_as_colliders=False)

        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf"),
            builder,
            xform=wp.transform(np.array((0.0, 0.05, 0.05)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            parse_visuals_as_colliders=False)

        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf"),
            builder,
            xform=wp.transform(np.array((0.01, 0.15, 0.03)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            parse_visuals_as_colliders=False)
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/isaacgymenvs/objects/cube_multicolor_allegro.urdf"),
            builder,
            xform=wp.transform(np.array((0.01, 0.05, 0.13)), wp.quat_identity()),
            floating=True,
            density=1e2,  # use inertia settings from URDF
            armature=0.0,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+3,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            parse_visuals_as_colliders=False)

    def before_simulate(self):
        # apply some motion to the hand
        body_qd = self.state.body_qd.numpy()
        for i in range(self.num_envs):
            body_qd[i*self.bodies_per_env][2] = 0.4
            body_qd[i*self.bodies_per_env][1] = 0.2
        self.state.body_qd = wp.array(body_qd, dtype=wp.spatial_vector, device=self.device)

if __name__ == "__main__":
    run_demo(Demo)