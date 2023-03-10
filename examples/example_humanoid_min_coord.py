# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Humanoid Min Coord
#
# Compute the minimum y coordinate for multiple articulated humanoid models
# in parallel and visualized the results in a USD file.
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
from warp.sim import ModelShapeGeometry
from warp.sim.collide import get_box_vertex

from env.environment import compute_env_offsets

wp.init()

@wp.kernel
def compute_min_coord(
    coord: int,
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geo: ModelShapeGeometry,
    body_env: wp.array(dtype=int),
    # outputs
    min_coord: wp.array(dtype=float)
):
    shape = wp.tid()
    shape_type = geo.type[shape]
    body = shape_body[shape]
    env = body_env[body]
    scale = geo.scale[shape]

    val = min_coord[env]

    shape_tf = shape_X_bs[shape]
    if body >= 0:
        shape_tf = body_q[body] * shape_tf

    radius = scale[0]
    if shape_type == wp.sim.GEO_SPHERE:
        val = wp.transform_get_translation(shape_tf)[coord] - radius
    elif shape_type == wp.sim.GEO_BOX:
        val = wp.min(wp.transform_point(shape_tf, get_box_vertex(0, scale))[coord], val)
        val = wp.min(wp.transform_point(shape_tf, get_box_vertex(1, scale))[coord], val)
        val = wp.min(wp.transform_point(shape_tf, get_box_vertex(2, scale))[coord], val)
        val = wp.min(wp.transform_point(shape_tf, get_box_vertex(3, scale))[coord], val)
        val = wp.min(wp.transform_point(shape_tf, get_box_vertex(4, scale))[coord], val)
        val = wp.min(wp.transform_point(shape_tf, get_box_vertex(5, scale))[coord], val)
        val = wp.min(wp.transform_point(shape_tf, get_box_vertex(6, scale))[coord], val)
        val = wp.min(wp.transform_point(shape_tf, get_box_vertex(7, scale))[coord], val)
    elif shape_type == wp.sim.GEO_CAPSULE:
        half_height = scale[1]
        val = wp.min(wp.transform_point(shape_tf, wp.vec3(0.0, -half_height, 0.0))[coord]-radius, val)
        val = wp.min(wp.transform_point(shape_tf, wp.vec3(0.0, half_height, 0.0))[coord]-radius, val)

    wp.atomic_min(min_coord, env, val)


class Example:

    def __init__(self, stage=None, num_envs=1):

        builder = wp.sim.ModelBuilder()

        self.num_envs = num_envs

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "assets/nv_humanoid.xml"),
            articulation_builder)

        builder = wp.sim.ModelBuilder()

        spacing = 3.0

        offsets = compute_env_offsets(num_envs, env_offset=(spacing, 0.0, spacing))
        dof = len(articulation_builder.joint_q)

        body_env_ids = []

        for i in range(num_envs):
            pos = np.copy(offsets[i])
            # add some random height
            pos[1] = np.random.uniform(-1.0, 1.0)
            pos[2] -= 1.5
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(pos, wp.quat_from_axis_angle((1.0, 0.0, 0.0), np.random.uniform(0.0, np.pi)))
            )

            # random joint positions
            builder.joint_q[dof*i+7:] = np.random.uniform(-0.5, 0.5, dof-7)

            # mapping from body ID to humanoid (env) ID
            body_env_ids.extend([i] * articulation_builder.body_count)

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        #-----------------------
        # set up Usd renderer
        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=10.0)
            
        self.state = self.model.state()

        # evaluate body transforms from joint angles (forward kinematics)
        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)
        
        time = 0.0
        
        self.renderer.begin_frame(time)
        self.renderer.render(self.state)

        # compute minimal y coordinates per humanoid
        min_y = wp.array(np.ones(num_envs)*1000.0, dtype=wp.float32)
        body_env_ids = wp.array(body_env_ids, dtype=wp.int32)
        min_coord = 1  # y axis
        wp.launch(
            compute_min_coord,
            dim=self.model.shape_count,
            inputs=[
                min_coord,
                self.state.body_q,
                self.model.shape_transform,
                self.model.shape_body,
                self.model.shape_geo,
                body_env_ids,
            ],
            outputs=[min_y])
        
        min_y = min_y.numpy()

        for i in range(num_envs):
            pos = offsets[i]
            pos[min_coord] = min_y[i]
            self.renderer.render_box(f"min_y_{i}", pos=pos, rot=(0.,0.,0.,1.), extents=(spacing*0.35, 0.01, spacing*0.35))

        self.renderer.end_frame()

        self.renderer.save()


stage = os.path.join(os.path.dirname(__file__), "outputs/example_humanoid_min_coord.usd")
robot = Example(stage, num_envs=100)
