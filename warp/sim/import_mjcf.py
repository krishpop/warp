# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import math
import os
import xml.etree.ElementTree as ET

import numpy as np

import warp as wp
from warp.sim.model import JOINT_COMPOUND, JOINT_UNIVERSAL
from warp.sim.model import Mesh


def parse_mjcf(
    filename,
    builder,
    xform=wp.transform(),
    density=1000.0,
    stiffness=0.0,
    damping=0.0,
    contact_ke=1000.0,
    contact_kd=100.0,
    contact_kf=100.0,
    contact_mu=0.5,
    contact_restitution=0.5,
    limit_ke=100.0,
    limit_kd=10.0,
    armature=0.0,
    armature_scale=1.0,
    parse_meshes=False,
    enable_self_collisions=True,
    up_axis="z",
):
    file = ET.parse(filename)
    root = file.getroot()

    if isinstance(up_axis, str):
        up_axis = "xyz".index(up_axis.lower())
    sqh = np.sqrt(0.5)
    if up_axis == 0:
        xform = wp.transform(xform.p, wp.quat(0.0, 0.0, -sqh, sqh) * xform.q)
    elif up_axis == 2:
        xform = wp.transform(xform.p, wp.quat(sqh, 0.0, 0.0, -sqh) * xform.q)

    use_degrees = True  # angles are in degrees by default
    euler_seq = [1, 2, 3]  # XYZ by default

    compiler = root.find("compiler")
    if compiler is not None:
        use_degrees = compiler.attrib.get("angle", "degree").lower() == "degree"
        euler_seq = ["xyz".index(c) + 1 for c in compiler.attrib.get("eulerseq", "xyz").lower()]

    def parse_float(node, key, default):
        if key in node.attrib:
            return float(node.attrib[key])
        else:
            return default

    def parse_vec(node, key, default):
        if key in node.attrib:
            return np.fromstring(node.attrib[key], sep=" ")
        else:
            return np.array(default)

    def parse_mesh(geom):
        import trimesh

        faces = []
        vertices = []
        stl_file = next(
            filter(
                lambda m: m.attrib["name"] == geom.attrib["mesh"],
                root.find("asset").findall("mesh"),
            )
        ).attrib["file"]
        # handle stl relative paths
        if not os.path.isabs(stl_file):
            stl_file = os.path.join(os.path.dirname(filename), stl_file)
        m = trimesh.load(stl_file)

        for v in m.vertices:
            vertices.append(np.array(v))

        for f in m.faces:
            faces.append(int(f[0]))
            faces.append(int(f[1]))
            faces.append(int(f[2]))
        return Mesh(vertices, faces), m.scale

    def parse_body(body, parent):
        body_name = body.attrib["name"]
        body_pos = parse_vec(body, "pos", (0.0, 0.0, 0.0))
        body_ori_euler = parse_vec(body, "euler", (0.0, 0.0, 0.0))
        if len(np.nonzero(body_ori_euler)[0]) > 0:
            if use_degrees:
                body_ori_euler *= np.pi / 180
            body_ori = wp.quat_from_euler(body_ori_euler, *euler_seq)
        else:
            body_ori = wp.quat_identity()
        if parent == -1:
            body_pos = wp.transform_point(xform, body_pos)
            body_ori = xform.q * body_ori

        joint_armature = []
        joint_name = []
        joint_pos = []

        linear_axes = []
        angular_axes = []
        joint_type = None

        joints = body.findall("joint")
        for i, joint in enumerate(joints):
            # default to hinge if not specified
            if "type" not in joint.attrib:
                joint.attrib["type"] = "hinge"

            joint_name.append(joint.attrib["name"])
            joint_pos.append(parse_vec(joint, "pos", (0.0, 0.0, 0.0)))
            # TODO parse joint (child transform) rotation?
            joint_range = parse_vec(joint, "range", (-3.0, 3.0))
            joint_armature.append(parse_float(joint, "armature", armature) * armature_scale)

            if joint.attrib["type"].lower() == "free":
                joint_type = wp.sim.JOINT_FREE
                break
            is_angular = joint.attrib["type"].lower() == "hinge"
            mode = wp.sim.JOINT_MODE_LIMIT
            if stiffness > 0.0 or "stiffness" in joint.attrib:
                mode = wp.sim.JOINT_MODE_TARGET_POSITION
            axis_vec = parse_vec(joint, "axis", (0.0, 0.0, 0.0))
            ax = wp.sim.model.JointAxis(
                axis=axis_vec,
                limit_lower=(np.deg2rad(joint_range[0]) if is_angular and use_degrees else joint_range[0]),
                limit_upper=(np.deg2rad(joint_range[1]) if is_angular and use_degrees else joint_range[1]),
                target_ke=parse_float(joint, "stiffness", stiffness),
                target_kd=parse_float(joint, "damping", damping),
                limit_ke=limit_ke,
                limit_kd=limit_kd,
                mode=mode,
            )
            if is_angular:
                angular_axes.append(ax)
            else:
                linear_axes.append(ax)

        link = builder.add_body(
            origin=wp.transform(body_pos, body_ori),  # will be evaluated in fk()
            armature=joint_armature[0],
            name=body_name,
        )

        if joint_type is None:
            if len(linear_axes) == 0:
                if len(angular_axes) == 0:
                    joint_type = wp.sim.JOINT_FIXED
                elif len(angular_axes) == 1:
                    joint_type = wp.sim.JOINT_REVOLUTE
                elif len(angular_axes) == 2:
                    joint_type = wp.sim.JOINT_UNIVERSAL
                elif len(angular_axes) == 3:
                    joint_type = wp.sim.JOINT_COMPOUND
            elif len(linear_axes) == 1 and len(angular_axes) == 0:
                joint_type = wp.sim.JOINT_PRISMATIC
            else:
                joint_type = wp.sim.JOINT_D6

        builder.add_joint(
            joint_type,
            parent,
            link,
            linear_axes,
            angular_axes,
            name="_".join(joint_name),
            parent_xform=wp.transform(body_pos + joint_pos[0], body_ori),
            child_xform=wp.transform(joint_pos[0], wp.quat_identity()),
        )

        # -----------------
        # add shapes

        for geom in body.findall("geom"):
            geom_name = geom.attrib["name"]
            geom_type = geom.attrib["type"]

            geom_size = parse_vec(geom, "size", [1.0])
            geom_pos = parse_vec(geom, "pos", (0.0, 0.0, 0.0))
            geom_rot = wp.quat(*parse_vec(geom, "quat", (0.0, 0.0, 0.0, 1.0)))
            geom_density = parse_float(geom, "density", density)

            if geom_type == "sphere":
                builder.add_shape_sphere(
                    link,
                    pos=geom_pos,
                    rot=geom_rot,
                    radius=geom_size[0],
                    density=geom_density,
                    ke=contact_ke,
                    kd=contact_kd,
                    kf=contact_kf,
                    mu=contact_mu,
                    restitution=contact_restitution,
                )

            elif geom_type == "mesh" and parse_meshes:
                mesh, scale = parse_mesh(geom)
                geom_size = tuple([scale * s for s in geom_size])
                assert len(geom_size) == 3, "need to specify size for mesh geom"
                builder.add_shape_mesh(
                    body=link,
                    pos=geom_pos,
                    rot=geom_rot,
                    mesh=mesh,
                    scale=geom_size,
                    density=density,
                    ke=contact_ke,
                    kd=contact_kd,
                    kf=contact_kf,
                    mu=contact_mu,
                )

            elif geom_type in {"capsule", "cylinder"}:
                if "fromto" in geom.attrib:
                    geom_fromto = parse_vec(geom, "fromto", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

                    start = geom_fromto[0:3]
                    end = geom_fromto[3:6]

                    # compute rotation to align the Warp capsule (along x-axis), with mjcf fromto direction
                    axis = wp.normalize(end - start)
                    angle = math.acos(np.dot(axis, (0.0, 1.0, 0.0)))
                    axis = wp.normalize(np.cross(axis, (0.0, 1.0, 0.0)))

                    geom_pos = (start + end) * 0.5
                    geom_rot = wp.quat_from_axis_angle(axis, -angle)

                    geom_radius = geom_size[0]
                    geom_height = np.linalg.norm(end - start) * 0.5
                    geom_up_axis = 1

                else:
                    geom_radius = geom_size[0]
                    geom_height = geom_size[1]
                    geom_up_axis = up_axis

                if geom_type == "cylinder":
                    builder.add_shape_cylinder(
                        link,
                        pos=geom_pos,
                        rot=geom_rot,
                        radius=geom_radius,
                        half_height=geom_height,
                        density=density,
                        ke=contact_ke,
                        kd=contact_kd,
                        kf=contact_kf,
                        mu=contact_mu,
                        restitution=contact_restitution,
                        up_axis=geom_up_axis,
                    )
                else:
                    builder.add_shape_capsule(
                        link,
                        pos=geom_pos,
                        rot=geom_rot,
                        radius=geom_radius,
                        half_height=geom_height,
                        density=density,
                        ke=contact_ke,
                        kd=contact_kd,
                        kf=contact_kf,
                        mu=contact_mu,
                        restitution=contact_restitution,
                        up_axis=geom_up_axis,
                    )

            else:
                print("MJCF parsing issue: geom type", geom_type, "is unsupported")

        # -----------------
        # recurse

        for child in body.findall("body"):
            parse_body(child, link)

    # -----------------
    # start articulation

    start_shape_count = len(builder.shape_geo_type)
    builder.add_articulation()

    world = root.find("worldbody")
    for body in world.findall("body"):
        parse_body(body, -1)

    end_shape_count = len(builder.shape_geo_type)

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.add((i, j))
