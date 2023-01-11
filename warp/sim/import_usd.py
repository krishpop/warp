# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
import re

import warp as wp
from . import ModelBuilder

def parse_usd(
    filename, 
    builder: ModelBuilder, 
    default_density=1.0e3,
    only_load_enabled_rigid_bodies=False,
    only_load_enabled_joints=True,
    default_ke=1e5,
    default_kd=250.0,
    default_kf=500.0,
    default_mu=0.0,
    default_restitution=0.0,
    default_contact_thickness=0.0,
    verbose=True):

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, Sdf, Ar
    except ImportError:
        raise ImportError("Failed to import pxr. Please install USD.")
    
    def get_attribute(prim, name):
        if "*" in name:
            regex = name.replace('*', '.*')
            for attr in prim.GetAttributes():
                if re.match(regex, attr.GetName()):
                    return attr
        else:
            return prim.GetAttribute(name)

    def parse_float(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr:
            return default
        val = attr.Get()
        if np.isfinite(val):
            return val
        return default

    def parse_quat(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr:
            return default
        val = attr.Get()
        quat = wp.quat(*val.imaginary, val.real)
        l = wp.length(quat)
        if np.isfinite(l) and l > 0.0:
            return quat
        return default

    def parse_vec(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr:
            return default
        val = attr.Get()
        if np.isfinite(val).all():
            return np.array(val)
        return default

    def parse_generic(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr:
            return default
        return attr.Get()

    def str2axis(s: str) -> np.ndarray:
        axis = np.zeros(3)
        axis["XYZ".index(s.upper())] = 1.0
        return axis

    stage = Usd.Stage.Open(filename, Usd.Stage.LoadAll)
    if UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage):
        mass_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    else:
        mass_unit = 1.0
    if UsdGeom.StageHasAuthoredMetersPerUnit(stage):
        linear_unit = UsdGeom.GetStageMetersPerUnit(stage)
    else:
        linear_unit = 1.0


    def parse_xform(prim):
        xform = UsdGeom.Xform(prim)
        mat = np.array(xform.GetLocalTransformation())
        rot = wp.quat_from_matrix(mat[:3,:3])
        pos = mat[3, :3] * linear_unit
        scale = np.ones(3)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale = np.array(op.Get())
        return wp.transform(pos, rot), scale

    def parse_axis(prim, type, joint_data, is_angular, axis=None):
        # parse joint axis data
        schemas = prim.GetAppliedSchemas()  # TODO consider inherited schemas
        schemas_str = "".join(schemas)
        if f"DriveAPI:{type}" not in schemas_str and f"PhysicsLimitAPI:{type}" not in schemas_str:
            return
        drive_type = parse_generic(prim, f"drive:{type}:physics:type", "force")
        if drive_type != "force":
            print(f"Warning: only force drive type is supported, ignoring drive:{type} for joint {path}")
            return
        stiffness = parse_float(prim, f"drive:{type}:physics:stiffness", 0.0)
        damping = parse_float(prim, f"drive:{type}:physics:damping", 0.0)
        low = parse_float(prim, f"limit:{type}:physics:low")
        high = parse_float(prim, f"limit:{type}:physics:high")
        target_pos = parse_float(prim, f"drive:{type}:physics:targetPosition")
        target_vel = parse_float(prim, f"drive:{type}:physics:targetVelocity")
        if is_angular:
            stiffness *= mass_unit * linear_unit**2
            stiffness = np.deg2rad(stiffness)
            damping *= mass_unit * linear_unit**2
            damping = np.deg2rad(damping)
            if target_pos is not None:
                target_pos = np.deg2rad(target_pos)
            if target_vel is not None:
                target_vel = np.deg2rad(target_vel)
            if low is None:
                low = joint_data["lowerLimit"]
            else:
                low = np.deg2rad(low)
            if high is None:
                high = joint_data["upperLimit"]
            else:
                high = np.deg2rad(high)
        else:
            stiffness *= mass_unit
            damping *= mass_unit
            if target_pos is not None:
                target_pos *= linear_unit
            if target_vel is not None:
                target_vel *= linear_unit
            if low is None:
                low = joint_data["lowerLimit"]
            else:
                low *= linear_unit
            if high is None:
                high = joint_data["upperLimit"]
            else:
                high *= linear_unit
        mode = wp.sim.JOINT_MODE_LIMIT
        if f"DriveAPI:{type}" in schemas_str:
            if target_vel is not None:
                mode = wp.sim.JOINT_MODE_TARGET_VELOCITY
            else:
                mode = wp.sim.JOINT_MODE_TARGET_POSITION
        axis = wp.sim.JointAxis(
            axis=(axis or joint_data["axis"]),
            lower_limit=low,
            upper_limit=high,
            target=(target_pos or target_vel or (low + high) / 2),
            target_ke=stiffness,
            target_kd=damping,
            mode=mode,
        )
        if is_angular:
            joint_data["angular_axes"].append(axis)
        else:
            joint_data["linear_axes"].append(axis)

    upaxis = str2axis(UsdGeom.GetStageUpAxis(stage))

    shape_types = {"Cube", "Sphere", "Mesh", "Capsule", "Plane", "Cylinder", "Cone"}

    path_body_map = {}
    path_shape_map = {}
    # maps prim path name to its world transform
    prim_poses = {}
    # transform from body frame to where the actual joint child frame is
    # so that the link's children will use the right parent tf for the joint
    # prim_joint_xforms = {}
    path_collision_filters = set()
    no_collision_shapes = set()

    # first find all joints and materials
    joint_data = {}  # mapping from path of child link to joint USD settings
    materials = {}  # mapping from material path to material USD settings
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName())
        path = str(prim.GetPath())
        if verbose:
            print(path, type_name)
        if type_name.endswith("Joint"):
            # the type name can sometimes be "DistancePhysicsJoint" or "PhysicsDistanceJoint" ...
            type_name = type_name.replace("Physics", "").replace("Joint", "")
            child = str(prim.GetRelationship("physics:body1").GetTargets()[0])
            pos0 = parse_vec(prim, "physics:localPos0", np.zeros(3)) * linear_unit
            pos1 = parse_vec(prim, "physics:localPos1", np.zeros(3)) * linear_unit
            rot0 = parse_quat(prim, "physics:localRot0", wp.quat_identity())
            rot1 = parse_quat(prim, "physics:localRot1", wp.quat_identity())
            lower = parse_float(prim, "physics:lowerLimit", -np.inf)
            upper = parse_float(prim, "physics:upperLimit", np.inf)
            joint_data[child] = {
                "type": type_name,
                "name": str(prim.GetName()),
                "parent_tf": wp.transform(pos0, rot0),
                "child_tf": wp.transform(pos1, rot1),
                "enabled": parse_generic(prim, "physics:jointEnabled", True),
                "collisionEnabled": parse_generic(prim, "physics:collisionEnabled", False),
                "excludeFromArticulation": parse_generic(prim, "physics:excludeFromArticulation", False),
                "axis": str2axis(parse_generic(prim, "physics:axis", "X")),
                "breakForce": parse_float(prim, "physics:breakForce", np.inf),
                "breakTorque": parse_float(prim, "physics:breakTorque", np.inf),
                "linear_axes": [],
                "angular_axes": [],
            }
            if only_load_enabled_joints and not joint_data[child]["enabled"]:
                print("Skipping disabled joint", path)
                continue
            # parse joint limits
            joint_data[child]["lowerLimit"] = -np.inf
            joint_data[child]["upperLimit"] = np.inf
            if type_name == "Distance":
                # if distance is negative the joint is not limited
                joint_data[child]["lowerLimit"] = parse_float(prim, "physics:minDistance", -1.0) * linear_unit
                joint_data[child]["upperLimit"] = parse_float(prim, "physics:maxDistance", -1.0) * linear_unit
            elif type_name == "Prismatic":
                joint_data[child]["lowerLimit"] = lower * linear_unit
                joint_data[child]["upperLimit"] = upper * linear_unit
            else:
                joint_data[child]["lowerLimit"] = np.deg2rad(lower) if np.isfinite(lower) else lower
                joint_data[child]["upperLimit"] = np.deg2rad(upper) if np.isfinite(upper) else upper
            parents = prim.GetRelationship("physics:body0").GetTargets()
            if len(parents) > 0:
                joint_data[child]["parent"] = str(parents[0])
            else:
                joint_data[child]["parent"] = None
            
            # parse joint drive
            parse_axis(prim, "angular", joint_data[child], is_angular=True)
            parse_axis(prim, "rotX", joint_data[child], is_angular=True, axis=(1.0, 0.0, 0.0))
            parse_axis(prim, "rotY", joint_data[child], is_angular=True, axis=(0.0, 1.0, 0.0))
            parse_axis(prim, "rotZ", joint_data[child], is_angular=True, axis=(0.0, 0.0, 1.0))
            parse_axis(prim, "linear", joint_data[child], is_angular=False)
            parse_axis(prim, "transX", joint_data[child], is_angular=False, axis=(1.0, 0.0, 0.0))
            parse_axis(prim, "transY", joint_data[child], is_angular=False, axis=(0.0, 1.0, 0.0))
            parse_axis(prim, "transZ", joint_data[child], is_angular=False, axis=(0.0, 0.0, 1.0))

        elif type_name == "Material":
            material = {}
            if prim.HasAttribute("physics:density"):
                material["density"] = parse_float(prim, "physics:density", 0.0) * mass_unit / (linear_unit**3)
            if prim.HasAttribute("physics:restitution"):
                material["restitution"] = parse_float(prim, "physics:restitution", default_restitution)
            if prim.HasAttribute("physics:staticFriction"):
                material["staticFriction"] = parse_float(prim, "physics:staticFriction", default_mu)
            if prim.HasAttribute("physics:dynamicFriction"):
                material["dynamicFriction"] = parse_float(prim, "physics:dynamicFriction", default_mu)
            materials[path] = material
            
        elif type_name == "PhysicsScene":
            scene = UsdPhysics.Scene(prim)
            g_vec = scene.GetGravityDirectionAttr()
            g_mag = scene.GetGravityMagnitudeAttr()
            if g_mag.HasAuthoredValue() and np.isfinite(g_mag.Get()):
                builder.gravity = g_mag.Get() * linear_unit
            if g_vec.HasAuthoredValue() and np.linalg.norm(g_vec.Get()) > 0.0:
                builder.upvector = np.array(g_vec.Get())  # TODO flip sign?
            else:
                builder.upvector = upaxis

    def parse_prim(prim, incoming_xform, incoming_scale, incoming_schemas=[]):
        nonlocal builder
        nonlocal joint_data
        nonlocal path_body_map
        nonlocal path_shape_map
        nonlocal prim_poses
        # nonlocal prim_joint_xforms
        nonlocal path_collision_filters
        nonlocal no_collision_shapes

        path = str(prim.GetPath())
        type_name = str(prim.GetTypeName())
        schemas = set(prim.GetAppliedSchemas() + list(incoming_schemas))
        if verbose:
            print(path, type_name)
        children_refs = prim.GetChildren()

        # prim_joint_xforms[path] = wp.transform_identity()
        
        if type_name == "Xform":
            xform, scale = parse_xform(prim)
            xform = wp.mul(incoming_xform, xform)
            prim_poses[path] = xform
            # TODO support instancing of shapes in Warp.sim
            if prim.IsInstance():
                proto = prim.GetPrototype()
                for child in proto.GetChildren():
                    parse_prim(child, xform, incoming_scale*scale, schemas)
            else:
                for child in children_refs:
                    parse_prim(child, xform, incoming_scale*scale, schemas)
        elif type_name == "Scope":
            for child in children_refs:
                parse_prim(child, incoming_xform, incoming_scale, schemas)
        elif type_name in shape_types:
            if path in joint_data:
                joint = joint_data[path]
            else:
                joint = None

            shape_params = dict(
                ke=default_ke, kd=default_kd, kf=default_kf, mu=default_mu,
                restitution=default_restitution)

            density = None

            material = None
            if prim.HasRelationship("material:binding:physics"):
                other_paths = prim.GetRelationship("material:binding:physics").GetTargets()
                if len(other_paths) > 0:
                    material = materials[str(other_paths[0])]
            if material is not None:
                if "restitution" in material:
                    shape_params["restitution"] = material["restitution"]
                if "dynamicFriction" in material:
                    shape_params["mu"] = material["dynamicFriction"]
                if "density" in material:
                    density = material["density"]

            # assert prim.GetAttribute('orientation').Get() == "rightHanded", "Only right-handed orientations are supported."
            enabled = parse_generic(prim, "physics:rigidBodyEnabled", True)
            if only_load_enabled_rigid_bodies and not enabled:
                if verbose:
                    print("Skipping disabled rigid body", path)
                return
            mass = parse_float(prim, "physics:mass")
            if "PhysicsRigidBodyAPI" in schemas:
                if prim.HasAttribute("physics:density"):
                    d = parse_float(prim, "physics:density")
                    if d > 0.0:
                        density = d * mass_unit / (linear_unit**3)
            else:
                density = 0.0  # static object
            if density is None:
                density = default_density
            com = parse_vec(prim, "physics:centerOfMass", np.zeros(3))
            i_diag = parse_vec(prim, "physics:diagonalInertia", np.zeros(3))
            i_rot = parse_quat(prim, "physics:principalAxes", wp.quat_identity())

            xform, scale = parse_xform(prim)
            scale = incoming_scale*scale
            xform = wp.mul(incoming_xform, xform)
            prim_poses[path] = xform
            
            geo_tf = wp.transform_identity()
            

            body_id = builder.add_body(
                com=com,
                body_name=prim.GetName(),
                origin=wp.transform_identity(),
            )

            if joint is not None:
                joint_params = dict(
                    child=body_id,
                    linear_axes=joint["linear_axes"],
                    angular_axes=joint["angular_axes"],
                    name=joint["name"],
                    enabled=joint["enabled"],
                    parent_xform=joint["parent_tf"],
                    child_xform=joint["child_tf"])
                if joint["type"] == "Revolute":
                    joint_params["joint_type"] = wp.sim.JOINT_REVOLUTE
                    # if "drive:angular" in joint:
                    #     joint_params["joint_target_ke"] = joint["drive:angular"]["stiffness"]
                    #     joint_params["joint_target_kd"] = joint["drive:angular"]["damping"]
                    #     joint_params["joint_target"] = joint["drive:angular"]["target_pos"]
                elif joint["type"] == "Prismatic":
                    joint_params["joint_type"] = wp.sim.JOINT_PRISMATIC
                    # if "drive:linear" in joint:
                    #     joint_params["joint_target_ke"] = joint["drive:linear"]["stiffness"]
                    #     joint_params["joint_target_kd"] = joint["drive:linear"]["damping"]
                    #     joint_params["joint_target"] = joint["drive:linear"]["target_pos"]
                elif joint["type"] == "Spherical":
                    joint_params["joint_type"] = wp.sim.JOINT_BALL
                elif joint["type"] == "Fixed":
                    joint_params["joint_type"] = wp.sim.JOINT_FIXED
                elif joint["type"] == "Distance":
                    joint_params["joint_type"] = wp.sim.JOINT_DISTANCE
                elif joint["type"] == "":
                    joint_params["joint_type"] = wp.sim.JOINT_D6
                else:
                    print(f"Warning: unsupported joint type {joint['type']} for {path}")
                # joint_params["joint_axis"] = joint["axis"]
                # joint_params["joint_limit_lower"] = joint["lowerLimit"]
                # joint_params["joint_limit_upper"] = joint["upperLimit"]

                # joint_params["joint_xform"] = joint["parent_tf"]
                if joint["parent"] is None:
                    joint_params["parent"] = -1
                    # rel_pose = wp.transform_identity()
                else:
                    joint_params["parent"] = path_body_map[joint["parent"]]
                    # X_wp = prim_poses[joint["parent"]]
                    # X_wc = xform
                    # rel_pose = wp.transform_inverse(X_wp) * X_wc
                    # joint_params["joint_xform"] *= prim_joint_xforms[joint["parent"]]
                # joint_params["joint_xform"] = rel_pose
                # joint_params["joint_xform"] = wp.mul(joint["parent_tf"], joint["child_tf"])
                # joint_params["joint_xform_child"] = wp.transform_inverse(joint["child_tf"])
                # joint_params["joint_xform_child"] = wp.transform(-np.array(joint["child_tf"].p), joint["child_tf"].q)
                # joint_params["joint_xform_child"] = joint["child_tf"]
                # XXX apply child transform to shape since joint_xform_child is reserved for multi-dof joints
                # geo_tf = joint["child_tf"]
                # geo_tf = rel_pose * joint["child_tf"]
                # update relative transform for child prims
                # prim_joint_xforms[path] = geo_tf
                # geo_tf = wp.transform(-np.array(joint["child_tf"].p), joint["child_tf"].q)
                builder.add_joint(**joint_params)
            is_static = False
            if "PhysicsRigidBodyAPI" not in schemas:
                is_static = True
                density = 0.0
                builder.body_q[-1] = xform
            if joint is None and "PhysicsRigidBodyAPI" in schemas:
                builder.add_joint_free(child=body_id)
                # free joint; we set joint_q/qd, not body_q/qd since eval_fk is used after model creation
                builder.joint_q[-4:] = xform.q
                builder.joint_q[-7:-4] = xform.p
                linear_vel = parse_vec(prim, "physics:velocity", np.zeros(3)) * linear_unit
                angular_vel = parse_vec(prim, "physics:angularVelocity", np.zeros(3)) * linear_unit
                builder.joint_qd[-6:-3] = angular_vel
                builder.joint_qd[-3:] = linear_vel

            if prim.HasAttribute("doubleSided") and not prim.GetAttribute("doubleSided").Get():
                print(f"Warning: treating {path} as double-sided because single-sided collisions are not supported.")

            if type_name == "Cube":
                size = parse_float(prim, "size", 2.0)
                if prim.HasAttribute("extents"):
                    extents = parse_vec(prim, "extents") * scale
                    # TODO position geom at extents center?
                    geo_pos = 0.5 * (extents[0] + extents[1])
                    extents = extents[1] - extents[0]
                    print("extents", extents)
                else:
                    extents = scale * size
                shape_id = builder.add_shape_box(
                    body_id, geo_tf.p, geo_tf.q,
                    hx=extents[0]/2, hy=extents[1]/2, hz=extents[2]/2,
                    density=density, contact_thickness=default_contact_thickness,
                    **shape_params)
            elif type_name == "Sphere":
                if not (scale[0] == scale[1] == scale[2]):
                    print("Warning: Non-uniform scaling of spheres is not supported.")
                if prim.HasAttribute("extents"):
                    extents = parse_vec(prim, "extents") * scale
                    # TODO position geom at extents center?
                    geo_pos = 0.5 * (extents[0] + extents[1])
                    extents = extents[1] - extents[0]
                    if not (extents[0] == extents[1] == extents[2]):
                        print("Warning: Non-uniform extents of spheres are not supported.")
                    radius = extents[0]
                else:
                    radius = parse_float(prim, "radius", 1.0) * scale[0]
                shape_id = builder.add_shape_sphere(
                    body_id, geo_tf.p, geo_tf.q,
                    radius, density=density,
                    **shape_params)
            elif type_name == "Plane":
                normal_str = parse_generic(prim, "axis", "Z").upper()
                geo_rot = geo_tf.q
                if normal_str != "Y":
                    normal = str2axis(normal_str)
                    c = np.cross(normal, (0.0, 1.0, 0.0))
                    angle = np.arcsin(np.linalg.norm(c))
                    axis = c / np.linalg.norm(c)
                    geo_rot = wp.mul(geo_rot, wp.quat_from_axis_angle(axis, angle))
                width = parse_float(prim, "width", 0.0) * scale[0]
                length = parse_float(prim, "length", 0.0) * scale[1]
                shape_id = builder.add_shape_plane(
                    body=body_id, pos=geo_tf.p, rot=geo_rot,
                    width=width, length=length,
                    contact_thickness=default_contact_thickness,
                    **shape_params)
            elif type_name == "Capsule":
                normal_str = parse_generic(prim, "axis", "Z").upper()
                geo_rot = geo_tf.q
                if normal_str != "X":
                    normal = str2axis(normal_str)
                    c = np.cross(normal, (1.0, 0.0, 0.0))
                    angle = np.arcsin(np.linalg.norm(c))
                    axis = c / np.linalg.norm(c)
                    geo_rot = wp.quat_from_axis_angle(axis, angle)
                radius = parse_float(prim, "radius", 0.5) * scale[0]
                length = parse_float(prim, "height", 2.0) / 2 * scale[1]
                assert not prim.HasAttribute("extents"), "Capsule extents are not supported."
                shape_id = builder.add_shape_capsule(
                    body_id, geo_tf.p, geo_rot,
                    radius, length, density=density,
                    **shape_params)
            elif type_name == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                points = np.array(mesh.GetPointsAttr().Get())
                indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
                counts = mesh.GetFaceVertexCountsAttr().Get()
                faces = []
                face_id = 0
                for count in counts:
                    if count == 3:
                        faces.append(indices[face_id:face_id+3])
                    elif count == 4:
                        faces.append(indices[face_id:face_id+3])
                        faces.append(indices[[face_id,face_id+2,face_id+3]])
                    else:
                        # assert False, f"Error while parsing USD mesh {path}: encountered polygon with {count} vertices, but only triangles and quads are supported."
                        continue
                    face_id += count
                m = wp.sim.Mesh(points, np.array(faces).flatten())
                shape_id = builder.add_shape_mesh(
                    body_id, geo_tf.p, geo_tf.q,
                    scale=scale, mesh=m, density=density, contact_thickness=default_contact_thickness,
                    **shape_params)
            else:
                print(f"Warning: Unsupported geometry type {type_name} at {path}.")
                return

            path_body_map[path] = body_id
            path_shape_map[path] = shape_id
            com = parse_vec(prim, "physics:centerOfMass")
            if com is not None:
                # overwrite COM
                builder.body_com[body_id] = com * scale

            if prim.HasRelationship("physics:filteredPairs"):
                other_paths = prim.GetRelationship("physics:filteredPairs").GetTargets()
                for other_path in other_paths:
                    path_collision_filters.add((path, str(other_path)))

            if "PhysicsCollisionAPI" not in schemas:
                no_collision_shapes.add(shape_id)

            if mass is not None and not ("PhysicsRigidBodyAPI" in schemas and mass == 0.0):
                mass_ratio = mass / builder.body_mass[body_id]
                # mass has precedence over density, so we overwrite the mass computed from density
                builder.body_mass[body_id] = mass * mass_unit
                if mass > 0.0:
                    builder.body_inv_mass[body_id] = 1.0 / builder.body_mass[body_id]
                else:
                    builder.body_inv_mass[body_id] = 0.0
                # update inertia
                builder.body_inertia[body_id] *= mass_ratio
                if builder.body_inertia[body_id].any():
                    builder.body_inv_inertia[body_id] = np.linalg.inv(builder.body_inertia[body_id])
                else:
                    builder.body_inv_inertia[body_id] = np.zeros((3, 3))

            if np.linalg.norm(i_diag) > 0.0:
                rot = np.array(wp.quat_to_matrix(i_rot)).reshape(3, 3)
                inertia = rot @ np.diag(i_diag) @ rot.T
                builder.body_inertia[body_id] = inertia
                if inertia.any():
                    builder.body_inv_inertia[body_id] = np.linalg.inv(inertia)
                else:
                    builder.body_inv_inertia[body_id] = np.zeros((3, 3))

        elif type_name.endswith("Joint") or type_name.endswith("Light") or type_name.endswith("Scene") or type_name.endswith("Material"):
            return
        else:
            print(f"Warning: encountered unsupported prim type {type_name}")

    parse_prim(
        stage.GetDefaultPrim(),
        incoming_xform=wp.transform_identity(),
        incoming_scale=np.ones(3) * linear_unit)

    shape_count = len(builder.shape_geo_type)

    # apply collision filters now that we have added all shapes
    for path1, path2 in path_collision_filters:
        shape1 = path_shape_map[path1]
        shape2 = path_shape_map[path2]
        builder.shape_collision_filter_pairs.add((shape1, shape2))

    # apply collision filters to all shapes that have no collision
    for shape_id in no_collision_shapes:
        for other_shape_id in range(shape_count):
            if other_shape_id != shape_id:
                builder.shape_collision_filter_pairs.add((shape_id, other_shape_id))

    # return timing parameters
    return {
        "fps": stage.GetFramesPerSecond(),
        "duration": stage.GetEndTimeCode() - stage.GetStartTimeCode(),
        "upaxis": UsdGeom.GetStageUpAxis(stage).lower()
    }