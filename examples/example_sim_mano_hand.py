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

import skinning

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


@wp.kernel
def skin_mesh(
    body_q: wp.array(dtype=wp.transform),
    orig_inv_body_q: wp.array(dtype=wp.transform),
    vertices: wp.array(dtype=wp.vec3),
    orig_vertices: wp.array(dtype=wp.vec3),
    max_bones_per_vertex: int,
    skinning_ids: wp.array(dtype=int),
    skinning_weights: wp.array(dtype=float)
):
    vert_nr = wp.tid()

    v_orig = orig_vertices[vert_nr]
    v_sum = wp.vec3(0.0, 0.0, 0.0)
    w_sum = float(0.0)
    first = vert_nr * max_bones_per_vertex

    for i in range(max_bones_per_vertex):
        id = skinning_ids[first + i]
        w = skinning_weights[first + i]
        # TODO transform vector or point?
        # Matthias:  v_sum = v_sum + body_q[id] * orig_inv_body_q[id] * v_orig * w
        tf = body_q[id] * orig_inv_body_q[id]
        v_sum = v_sum + wp.transform_point(tf, v_orig) * w
        w_sum = w_sum + w

    if w_sum > 0.0:
        vertices[vert_nr] = v_sum / w_sum


class Example:
    frame_dt = 1.0 / 60.0

    episode_duration = 120.0  # seconds
    episode_frames = int(episode_duration / frame_dt)

    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)

    sim_time = 0.0

    def __init__(self, stage=None, render=True):

        self.enable_rendering = render
        self.device = wp.get_device("cuda")

        builder = wp.sim.ModelBuilder()

        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "assets/mano_hand/auto_mano_fixed_base.xml"),
            builder,
            xform=wp.transform((0.1, 0.0, 0.0), wp.quat_rpy(np.pi / 2, 0.0, 0.0)),
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

        # skinning

        self.max_bones_per_vertex = 4
        _, _, vertices, tri_ids = skinning.read_obj(os.path.join(
            os.path.dirname(__file__), "assets/mano_hand/mano_watertight.obj"))

        self.num_verts = len(vertices)
        self.vertices = wp.array(vertices, dtype=wp.vec3, device=self.device)
        self.orig_verts = wp.array(vertices, dtype=wp.vec3, device=self.device)
        self.tri_ids = wp.array(tri_ids, dtype=wp.vec3, device=self.device)

        self.skinned_mesh = wp.sim.Mesh(vertices, tri_ids)

        # builder.add_shape_mesh(
        #     body=-1,
        #     mesh=self.skinned_mesh,
        #     has_ground_collision=False,

        # )

        num_bones = len(builder.body_q)
        bone_transforms = []
        bone_shapes = []

        self.orig_inv_body_q = []

        for i in range(num_bones):
            h = 0.1
            for shape in builder.body_shapes[i]:
                h = 2.0 * builder.shape_geo_scale[shape][1]
            bone_shapes.append(wp.vec3(h, 0.0, 0.0))
            self.orig_inv_body_q.append(wp.transform_inverse(builder.body_q[i]))

            # TODO does this mean transform point or vector?
            # Matthias:  pos = wp.transform(builder.body_q[i], wp.vec3(-0.5 * h, 0.0, 0.0))
            pos = wp.transform_vector(builder.body_q[i], wp.vec3(-0.5 * h, 0.0, 0.0))
            mat = wp.quat_to_matrix(wp.transform_get_rotation(builder.body_q[i]))
            bone_transforms.append([
                [mat[0][0], mat[0][1], mat[0][2], pos[0]],
                [mat[1][0], mat[1][1], mat[1][2], pos[1]],
                [mat[2][0], mat[2][1], mat[2][2], pos[2]],
                [0.0, 0.0, 0.0, 1.0]])

        self.orig_inv_body_q = wp.array(self.orig_inv_body_q, dtype=wp.transform, device=self.device)

        skinning_ids, skinning_weights = skinning.compute_skinning_info(
            bone_transforms, bone_shapes, vertices, tri_ids, self.max_bones_per_vertex, resolution=30, device=self.device)

        self.skinning_ids = skinning_ids
        self.skinning_weights = skinning_weights

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.integrator = wp.sim.XPBDIntegrator()

        self.time_array = wp.zeros(1)

        self.renderer = None
        if render:
            self.renderer = wp.sim.render.SimRendererOpenGL(
                self.model, stage, show_joints=True, scaling=40.0, draw_axis=False)
            # self.renderer = wp.sim.render.SimRendererUsd(self.model, stage, scaling=500.0)

            self.renderer.render_mesh("hand_mesh", self.vertices.numpy(), self.tri_ids.numpy())

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
                    self.time_array],
                device=self.device)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        wp.launch(
            skin_mesh,
            dim=self.num_verts,
            inputs=[
                self.state_0.body_q, self.orig_inv_body_q,
                self.vertices, self.orig_verts,
                self.max_bones_per_vertex,
                self.skinning_ids, self.skinning_weights],
            device=self.device)

    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state_0)
        self.renderer.render_mesh("hand_mesh", self.vertices.numpy(), self.tri_ids.numpy())
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
