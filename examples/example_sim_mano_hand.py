# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim MANO Hand
#
# Shows how to set up a simulation of the MANO hand model
# from an MJCF using the wp.sim.ModelBuilder().
#
###########################################################################

import os

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


@wp.kernel
def animate_joints(
    dt: float,
    num_q: int,
    joint_lower: wp.array(dtype=float),
    joint_upper: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    time: wp.array(dtype=float),
):
    for i in range(num_q):
        p = 0.48 * wp.sin(time[0] + float(i) * 0.5) + 0.51
        joint_q[i] = (1.0 - p) * joint_lower[i] + p * joint_upper[i]

    wp.atomic_add(time, 0, dt)


class Example:
    frame_dt = 1.0 / 60.0

    episode_duration = 20.0  # seconds
    episode_frames = int(episode_duration / frame_dt)

    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)

    sim_time = 0.0

    def __init__(self, stage=None, render=True):
        builder = wp.sim.ModelBuilder()

        self.enable_rendering = render

        builder = wp.sim.ModelBuilder()

        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "assets/mano_hand/auto_mano_fixed_base.xml"),
            builder,
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_rpy(-np.pi / 2, np.pi * 0.75, np.pi / 2)),
            density=1e3,
            armature=0.1,
            stiffness=10.0,
            damping=0.0,
            scale=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False)

        # builder.plot_articulation()
        builder.collapse_fixed_joints()
        # builder.plot_articulation()

        # ensure all joint positions are within limits
        offset = 0  # skip floating base
        for i in range(len(builder.joint_limit_lower)):
            # builder.joint_q[i + offset] = 0.5 * \
            #     (builder.joint_limit_lower[i] + builder.joint_limit_upper[i])
            # builder.joint_target[i] = builder.joint_q[i + offset]
            builder.joint_target_ke[i] = 5000.0
            builder.joint_target_kd[i] = 1.0

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.integrator = wp.sim.XPBDIntegrator()

        self.time_array = wp.zeros(1)

        self.renderer = None
        if render:
            # self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, show_joints=True, scaling=40.0, draw_axis=False)
            self.renderer = wp.sim.render.SimRendererUsd(self.model, stage, scaling=500.0)

    def update(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            # wp.sim.collide(self.model, self.state_0)
            wp.launch(
                animate_joints,
                dim=1,
                inputs=[
                    self.sim_dt,
                    self.model.joint_axis_count,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.model.joint_target,
                    self.time_array]
            )
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()

    def run(self):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        profiler = {}

        # create update graph
        wp.capture_begin()

        # simulate
        self.update()

        graph = wp.capture_end()

        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):
            for f in range(0, self.episode_frames):
                with wp.ScopedTimer("simulate", active=True):
                    wp.capture_launch(graph)
                self.sim_time += self.frame_dt

                if self.enable_rendering:
                    with wp.ScopedTimer("render", active=True):
                        self.render()

            wp.synchronize()

        self.renderer.save()


if __name__ == "__main__":
    stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_mano_hand.usd")
    robot = Example(stage, render=True)
    robot.run()
