# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Joint Params
#
# Shows how to set up a chain of rigid bodies connected by revolute joints
# and animating the target position of the first joint.
#
###########################################################################

import numpy as np
import warp as wp
import warp.sim
import warp.sim.render

wp.init()

@wp.kernel
def modify_joint_params(
    time: float,
    joint_target: wp.array(dtype=float),
):
    # set the target position of the revolute joint
    tid = wp.tid()
    joint_target[tid] = wp.sin(time*2.0 + float(tid))*0.5


class Example:

    frame_dt = 1.0/60.0

    sim_substeps = 5
    sim_dt = frame_dt / sim_substeps
   
    sim_time = 0.0
    render_time = 0.0

    def __init__(self):

        chain_length = 3
        chain_width = 2.0
        joint_limit_lower=-np.deg2rad(120.0)
        joint_limit_upper=np.deg2rad(120.0)

        builder = wp.sim.ModelBuilder(gravity=0.0)


        # start a new articulation
        builder.add_articulation()

        for i in range(chain_length):

            if i == 0:
                parent = -1
                parent_joint_xform = wp.transform([0.0, 0.0, chain_width], wp.quat_identity())           
            else:
                parent = builder.joint_count-1
                parent_joint_xform = wp.transform([chain_width, 0.0, 0.0], wp.quat_identity())


            # create body
            b = builder.add_body(
                    origin=wp.transform([i, 0.0, chain_width], wp.quat_identity()),
                    armature=0.1)

            # create shape
            s = builder.add_shape_box( 
                    pos=(chain_width*0.5, 0.0, 0.0),
                    hx=chain_width*0.5,
                    hy=0.1,
                    hz=0.1,
                    density=10.0,
                    body=b)

            if i == 0:
                target_ke=1000.0
                target_kd=10.0
            else:
                target_ke=0.0
                target_kd=0.0

            builder.add_joint_revolute(
                parent=parent,
                child=b,
                axis=(0.0, 0.0, 1.0),
                parent_xform=parent_joint_xform,
                child_xform=wp.transform_identity(),
                limit_lower=joint_limit_lower,
                limit_upper=joint_limit_upper,
                target_ke=target_ke,
                target_kd=target_kd,
            )
            
        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.integrator = wp.sim.XPBDIntegrator(iterations=2)

        self.renderer = wp.sim.render.SimRendererNano(self.model, "Example Sim Joint Params", scaling=2.0)


    def update(self):
        # just animiate the first joint (dim=1 instead of model.joint_axis_count)
        wp.launch(
            modify_joint_params,
            dim=1,
            inputs=[self.sim_time],
            outputs=[self.model.joint_target])

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()

    def run(self):
        self.sim_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state_0)

        # simulate
        while True:
            self.update()
            self.render()

            wp.synchronize()
            self.sim_time += self.frame_dt


if __name__ == "__main__":
    robot = Example()
    robot.run()
