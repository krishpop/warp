# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import warp as wp
from warp.sim.model import Mesh


def parse_urdf(
        filename,
        builder,
        xform=wp.transform(),
        floating=False,
        density=1000.0,
        stiffness=100.0,
        damping=10.0,
        armature=0.0,
        shape_ke=1.e+4,
        shape_kd=1.e+3,
        shape_kf=1.e+2,
        shape_mu=0.25,
        shape_restitution=0.5,
        shape_thickness=0.0,
        limit_ke=100.0,
        limit_kd=10.0,
        parse_visuals_as_colliders=False,
        enable_self_collisions=True,
        ignore_inertial_definitions=False):

    import urdfpy
    # silence trimesh logging
    import logging
    logging.getLogger('trimesh').disabled = True

    robot = urdfpy.URDF.load(filename)

    # maps from link name -> link index
    link_index = {}

    builder.add_articulation()

    start_shape_count = len(builder.shape_geo_type)

    def parse_shapes(link, collisions, density):

        # add geometry
        for collision in collisions:

            origin = urdfpy.matrix_to_xyz_rpy(collision.origin)

            pos = origin[0:3]
            rot = wp.quatf(*wp.quat_rpy(*origin[3:6]))

            geo = collision.geometry

            if geo.box:
                builder.add_shape_box(
                    body=link,
                    pos=pos,
                    rot=rot,
                    hx=geo.box.size[0] * 0.5,
                    hy=geo.box.size[1] * 0.5,
                    hz=geo.box.size[2] * 0.5,
                    density=density,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu,
                    restitution=shape_restitution,
                    thickness=shape_thickness)

            if geo.sphere:
                builder.add_shape_sphere(
                    body=link,
                    pos=pos,
                    rot=rot,
                    radius=geo.sphere.radius,
                    density=density,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu,
                    restitution=shape_restitution,
                    thickness=shape_thickness)

            if geo.cylinder:

                builder.add_shape_capsule(
                    body=link,
                    pos=pos,
                    rot=rot,
                    radius=geo.cylinder.radius,
                    half_height=geo.cylinder.length * 0.5,
                    density=density,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu,
                    up_axis=2,  # cylinders in URDF are aligned with z-axis
                    restitution=shape_restitution,
                    thickness=shape_thickness)

            if geo.mesh:

                for m in geo.mesh.meshes:
                    faces = list(np.array(m.faces).astype('int').flatten())
                    vertices = np.array(m.vertices, dtype=np.float32).reshape((-1, 3))
                    if geo.mesh.scale is not None:
                        vertices *= geo.mesh.scale
                    mesh = Mesh(vertices, faces)
                    builder.add_shape_mesh(
                        body=link,
                        pos=pos,
                        rot=rot,
                        mesh=mesh,
                        density=density,
                        ke=shape_ke,
                        kd=shape_kd,
                        kf=shape_kf,
                        mu=shape_mu,
                        restitution=shape_restitution,
                        thickness=shape_thickness)

    # add links
    for urdf_link in robot.links:

        link = builder.add_body(
            origin=wp.transform_identity(),
            armature=armature,
            name=urdf_link.name)

        if parse_visuals_as_colliders:
            colliders = urdf_link.visuals
        else:
            colliders = urdf_link.collisions

        m = urdf_link.inertial.mass
        actual_density = 1.0 if m > 0.0 and density == 0.0 else density
        parse_shapes(link, colliders, density=actual_density)

        if not ignore_inertial_definitions and m > 0.0:
            # overwrite inertial parameters if defined
            com = urdfpy.matrix_to_xyz_rpy(urdf_link.inertial.origin)[0:3]
            I_m = urdf_link.inertial.inertia

            builder.body_mass[link] = m
            builder.body_inv_mass[link] = 1.0 / m
            builder.body_com[link] = com
            builder.body_inertia[link] = I_m
            builder.body_inv_inertia[link] = np.linalg.inv(I_m)

        # add ourselves to the index
        link_index[urdf_link.name] = link

    # add base joint
    root = link_index[robot.base_link.name]
    if floating:
        builder.add_joint_free(root, name="floating_base")

        # set dofs to transform
        start = builder.joint_q_start[root]

        builder.joint_q[start + 0] = xform.p[0]
        builder.joint_q[start + 1] = xform.p[1]
        builder.joint_q[start + 2] = xform.p[2]

        builder.joint_q[start + 3] = xform.q[0]
        builder.joint_q[start + 4] = xform.q[1]
        builder.joint_q[start + 5] = xform.q[2]
        builder.joint_q[start + 6] = xform.q[3]

    else:
        builder.add_joint_fixed(-1, root, parent_xform=xform, name="fixed_base")

    # add joints, in topological order starting from root body

    # find joints per body
    body_children = {link.name: [] for link in robot.links}
    # mapping from parent, child link names to joint
    parent_child_joint = {}

    for joint in robot.joints:
        body_children[joint.parent].append(joint.child)
        parent_child_joint[(joint.parent, joint.child)] = joint

    # topological sorting of joints because the FK solver will resolve body transforms
    # in joint order and needs the parent link transform to be resolved before the child
    visited = {link.name: False for link in robot.links}
    sorted_joints = []

    # depth-first search
    def dfs(joint):
        link = joint.child
        if visited[link]:
            return
        visited[link] = True

        for child in body_children[link]:
            if not visited[child]:
                dfs(parent_child_joint[(link, child)])

        sorted_joints.insert(0, joint)

    # start DFS from each unvisited joint
    for joint in robot.joints:
        if not visited[joint.parent]:
            dfs(joint)

    for joint in sorted_joints:
        parent = link_index[joint.parent]
        child = link_index[joint.child]

        origin = urdfpy.matrix_to_xyz_rpy(joint.origin)
        pos = origin[0:3]
        rot = wp.quat_rpy(*origin[3:6])

        lower = -1.0e3
        upper = 1.0e3
        joint_damping = damping

        # limits
        if joint.limit:
            if joint.limit.lower is not None:
                lower = joint.limit.lower
            if joint.limit.upper is not None:
                upper = joint.limit.upper

        # overwrite damping if defined in URDF
        if joint.dynamics:
            if joint.dynamics.damping:
                joint_damping = joint.dynamics.damping

        parent_xform = wp.transform(pos, rot)
        child_xform = wp.transform_identity()

        joint_mode = wp.sim.JOINT_MODE_LIMIT
        if stiffness > 0.0:
            joint_mode = wp.sim.JOINT_MODE_TARGET_POSITION

        joint_params = dict(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            name=joint.name,
        )

        if joint.joint_type == "revolute" or joint.joint_type == "continuous":
            builder.add_joint_revolute(
                axis=joint.axis,
                target_ke=stiffness, target_kd=joint_damping,
                limit_lower=lower, limit_upper=upper,
                limit_ke=limit_ke, limit_kd=limit_kd,
                mode=joint_mode,
                **joint_params)
        elif joint.joint_type == "prismatic":
            builder.add_joint_prismatic(
                axis=joint.axis,
                target_ke=stiffness, target_kd=joint_damping,
                limit_lower=lower, limit_upper=upper,
                limit_ke=limit_ke, limit_kd=limit_kd,
                mode=joint_mode,
                **joint_params)
        elif joint.joint_type == "fixed":
            builder.add_joint_fixed(**joint_params)
        elif joint.joint_type == "floating":
            builder.add_joint_free(**joint_params)
        elif joint.joint_type == "planar":
            # find plane vectors perpendicular to axis
            axis = np.array(joint.axis)
            axis /= np.linalg.norm(axis)

            # create helper vector that is not parallel to the axis
            helper = np.array([1, 0, 0]) if np.allclose(axis, [0, 1, 0]) else np.array([0, 1, 0])

            u = np.cross(helper, axis)
            u /= np.linalg.norm(u)

            v = np.cross(axis, u)
            v /= np.linalg.norm(v)

            builder.add_joint_d6(
                linear_axes=[
                    wp.sim.JointAxis(
                        u, limit_lower=lower, limit_upper=upper, limit_ke=limit_ke, limit_kd=limit_kd
                    ),
                    wp.sim.JointAxis(
                        v, limit_lower=lower, limit_upper=upper, limit_ke=limit_ke, limit_kd=limit_kd
                    ),
                ],
                **joint_params)
        else:
            raise Exception("Unsupported joint type: " + joint.joint_type)

    end_shape_count = len(builder.shape_geo_type)

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.add((i, j))
