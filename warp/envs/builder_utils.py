import os.path as osp
import os
import warp as wp
import math
import numpy as np
from pathlib import Path
from warp.envs.common import *
from warp.envs.tcdm_utils import get_tcdm_trajectory
from tcdm.envs import asset_abspath


def quat_multiply(a, b):
    return np.array(
        (
            a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - b[1] * a[2],
            a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - b[2] * a[0],
            a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - b[0] * a[1],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        )
    )


def add_joint(
    builder,
    pos,
    ori,
    joint_type,
    joint_axis=(0.0, 0.0, 1.0),
    body_name="object",
    limit_lower=-2 * np.pi * 3.0,
    limit_upper=2 * np.pi * 3.0,
    parent=-1,
    damping=0.75,
):
    """Add a joint to the builder"""

    body_link = builder.add_body(
        # parent=parent,
        origin=wp.transform(pos, wp.quat_identity()),
        name=body_name,
        armature=0.0,
    )
    if joint_type == wp.sim.JOINT_FREE:
        builder.add_joint_free(body_link, wp.transform(pos, ori), parent=parent, name="obj_joint")
    elif joint_type == wp.sim.JOINT_REVOLUTE:
        builder.add_joint_revolute(
            parent,
            body_link,
            wp.transform(pos, ori),
            wp.transform_identity(),
            axis=joint_axis,
            target_ke=0.0,  # 1.0
            target_kd=damping,  # 1.5
            limit_ke=100.0,
            limit_kd=150.0,
            limit_lower=limit_lower,
            limit_upper=limit_upper,
            name="obj_joint",
        )
    else:
        raise ValueError("Invalid joint")

    return body_link


def add_object(
    builder,
    link,
    object_type,
    rot=(1.0, 0.0, 0.0, 0.0),
    density=100.0,
    contact_ke=1e4,
    contact_kd=1e2,
    xform=None,
    scale=2,
    model_path=None,
):
    if object_type in OBJ_PATHS:
        add_mesh(builder, link, OBJ_PATHS[object_type], contact_ke=contact_ke, contact_kd=contact_kd, density=density)

    elif object_type in TCDM_TRAJ_PATHS:
        add_tcdm_mesh(
            builder,
            link,
            model_path,
            rot,
            density,
            contact_ke,
            contact_kd,
            1.3,
        )
    elif object_type is ObjectType.RECTANGULAR_PRISM_FLAT:
        add_box(
            builder,
            link,
            size=(0.1, 0.3, 0.1),
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
    elif object_type is ObjectType.RECTANGULAR_PRISM:
        add_box(
            builder,
            link,
            size=(0.15, 0.15, 0.2),
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
    elif object_type is ObjectType.CUBE:
        add_box(
            builder,
            link,
            size=(0.15, 0.15, 0.15),
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
    elif object_type is ObjectType.SPHERE:
        add_sphere(builder, link, 0.15, contact_ke, contact_kd)
    elif object_type is ObjectType.OCTPRISM:
        start_shape_count = len(builder.shape_geo_type)
        hy = 0.06  # dividing circumference into 8 segments
        hx = (math.sqrt(2) + 1) * hy  # length of box to extend to other side of prism
        hz = 0.15
        size = (hx * scale, hy * scale, hz * scale)
        add_box(
            builder,
            link,
            size=size,
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
        rot = wp.quat_from_axis_angle((0, 0, 1), math.pi / 4)
        add_box(
            builder,
            link,
            rot=rot,
            size=size,
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
        rot = wp.quat_from_axis_angle((0, 0, 1), math.pi / 2)
        add_box(
            builder,
            link,
            rot=rot,
            size=size,
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
        rot = wp.quat_from_axis_angle((0, 0, 1), 3 * math.pi / 4)
        add_box(
            builder,
            link,
            rot=rot,
            size=size,
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
        end_shape_count = len(builder.shape_geo_type)
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.add((i, j))


def add_mesh(builder, link, obj_path, ke=2.0e5, kd=1e4, density=100.0):
    obj_path = str(Path(osp.dirname(__file__)).parent / "envs" / "assets" / obj_path)
    mesh, scale = parse_mesh(obj_path)
    geom_size = (0.5, 0.5, 0.5)

    builder.add_shape_mesh(
        body=link,
        pos=(0.0, 0.0, 0.0),  # in Z-up frame, transform applied already
        # rot=(0.70710678, 0.0, 0.0, 0.70710678),
        mesh=mesh,
        scale=geom_size,
        density=density,
        ke=ke,
        kd=kd,
    )


def add_tcdm_mesh(
    builder,
    link,
    obj_path,
    rot=(1.0, 0.0, 0.0, 0.0),
    density=100.0,
    ke=2.0e5,
    kd=1e4,
    scale=1,
):
    #'hammer-use1' --> ['hammer', 'use1']
    obj_stl_path = asset_abspath(obj_path)
    mesh, _ = parse_mesh(obj_stl_path)
    geom_size = (scale, scale, scale)
    builder.add_shape_mesh(
        body=link,
        pos=(0.0, 0.0, 0.0),  # in Z-up frame, transform applied already
        rot=rot,
        mesh=mesh,
        scale=geom_size,
        density=density,
        ke=ke,
        kd=kd,
    )


def add_sphere(builder, link, radius=0.15, ke=1e5, kd=1e3):
    builder.add_shape_sphere(
        body=link,
        pos=(0.0, 0.0, 0.0),  # in Z-up frame, transform applied already
        radius=radius,
        density=100.0,
        ke=ke,
        kd=kd,
    )


def add_box(
    builder,
    link,
    rot=(0.0, 0.0, 0.0, 1.0),
    size=(0.25, 0.25, 0.25),
    density=100.0,
    ke=1e4,
    kd=100,
    xform=None,
):
    hx, hy, hz = size

    if xform is not None:
        q = wp.transform_get_rotation(xform)
        t = wp.transform_get_rotation(xform)
        rot = quat_multiply(q, rot)

    builder.add_shape_box(
        body=link,
        pos=(0.0, 0.0, hz),  # in Z-up frame, transform applied already
        rot=rot,
        hx=hx,
        hy=hy,
        hz=hz,
        density=density,
        ke=ke,
        kd=kd,
    )


def create_allegro_hand(builder, action_type, floating_base=False):
    stiffness, damping = 0.0, 0.0
    if action_type is ActionType.POSITION or action_type is ActionType.VARIABLE_STIFFNESS:
        stiffness, damping = 5000.0, 10.0
    if floating_base:
        xform = wp.transform(
            (0.01, 0.17, 0.125),
            # thumb up palm down
            # wp.quat_rpy(-np.pi / 2 * 3, np.pi * 1.25, np.pi / 2 * 3)
            # thumb up (default) palm orthogonal to gravity
            # wp.quat_rpy(-np.pi / 2 * 3, np.pi * 0.75, np.pi / 2 * 3),
            # thumb up, palm facing left
            # wp.quat_rpy(-np.pi / 2 * 3, np.pi * 0.75, np.pi / 2 * 3),
            # thumb up, palm facing right
            wp.quat_rpy(np.pi * 0.0, np.pi * 1.0, np.pi * -0.25),
        )
    else:
        xform = wp.transform(
            np.array((0.1, 0.15, 0.0)),
            # wp.quat_rpy(-np.pi / 2 * 3, np.pi * 0.75, np.pi / 2),  # thumb down
            # wp.quat_rpy(-np.pi / 2 * 3, np.pi * 0.75, np.pi / 2 * 3),  # thumb up (default) palm orthogonal to gravity
            wp.quat_rpy(-np.pi / 2 * 3, np.pi * 1.25, np.pi / 2 * 3),  # thumb up palm down
        )
    wp.sim.parse_urdf(
        os.path.join(
            os.path.split(os.path.dirname(__file__))[0],
            "envs/assets/isaacgymenvs/kuka_allegro_description/allegro.urdf",
        ),
        builder,
        xform=xform,
        floating=floating_base,
        density=1e3,
        armature=0.01,
        stiffness=stiffness,
        damping=damping,
        shape_ke=1.0e3,
        shape_kd=1.0e2,
        shape_kf=1.0e2,
        shape_mu=0.5,
        limit_ke=1.0e4,
        limit_kd=1.0e1,
        enable_self_collisions=False,
    )

    # ensure all joint positions are within limits
    q_offset = 7 if floating_base else 0
    for i in range(16):
        # if i > 17:
        #     builder.joint_q[i + q_offset] = builder.joint_limit_lower[i + qd_offset]
        # else:
        if floating_base and 12 <= i <= 13:
            x, y = 0.3, 0.7
        else:
            x, y = 0.65, 0.35
        builder.joint_q[i + q_offset] = x * (builder.joint_limit_lower[i]) + y * (builder.joint_limit_upper[i])
        builder.joint_target[i] = builder.joint_q[i + q_offset]

        if action_type is ActionType.POSITION or action_type is ActionType.VARIABLE_STIFFNESS:
            builder.joint_target_ke[i] = 5000.0
            builder.joint_target_kd[i] = 10.0
        else:
            builder.joint_target_ke[i] = 0.0
            builder.joint_target_kd[i] = 0.0


class ObjectModel:
    def __init__(
        self,
        object_type,
        base_pos=(0.0, 0.075, 0.0),
        base_ori=(np.pi / 2, 0.0, 0.0),
        joint_type=wp.sim.JOINT_FREE,
        contact_ke=1.0e3,
        contact_kd=100.0,
        scale=0.4,
        damping=0.5,
    ):
        self.object_type = object_type
        self.object_name = object_type.name.lower()
        self.base_pos = base_pos
        if len(base_ori) == 3:
            self.base_ori = tuple(x for x in wp.quat_rpy(*base_ori))
        elif len(base_ori) == 4:
            self.base_ori = base_ori
        self.joint_type = joint_type
        self.contact_ke = contact_ke
        self.contact_kd = contact_kd
        self.scale = scale
        self.damping = damping
        self.model_path = TCDM_MESH_PATHS.get(self.object_type)
        if self.model_path is not None:
            self.tcdm_trajectory, self.dex_trajectory = get_tcdm_trajectory(self.object_type)
        else:
            self.tcdm_trajectory = self.dex_trajectory = None

    def create_articulation(self, builder, density=100.0):
        self.object_joint = add_joint(
            builder,
            pos=self.base_pos,
            ori=self.base_ori,
            joint_type=self.joint_type,
            damping=self.damping,
            body_name="object",  # self.object_name + "_body_joint",
        )
        if self.model_path:
            obj_stl_path = asset_abspath(self.model_path)
            mesh, _ = parse_mesh(obj_stl_path)
            geom_size = (self.scale, self.scale, self.scale)
            builder.add_shape_mesh(
                body=self.object_joint,
                pos=(0.0, 0.0, 0.0),  # in Z-up frame, transform applied already
                rot=(1.0, 0.0, 0.0, 0.0),
                mesh=mesh,
                scale=geom_size,
                density=density,
                ke=self.contact_ke,
                kd=self.contact_kd,
            )
        else:
            add_object(
                builder,
                self.object_joint,
                self.object_type,
                self.base_ori,
                density,
                self.contact_ke,
                self.contact_kd,
                scale=self.scale,
            )


class OperableObjectModel(ObjectModel):
    def __init__(
        self,
        object_type,
        base_pos=(0.0, 0.075, 0.0),
        base_ori=(np.pi / 2, 0.0, 0.0),
        contact_ke=1.0e3,
        contact_kd=100.0,
        scale=0.4,
        damping=0.5,
        model_path: str = "",
    ):
        super().__init__(
            object_type,
            base_pos=base_pos,
            base_ori=base_ori,
            contact_ke=contact_ke,
            contact_kd=contact_kd,
            scale=scale,
            damping=damping,
        )
        assert model_path and os.path.splitext(model_path)[1] == ".urdf"
        self.model_path = model_path

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets", self.model_path),
            builder,
            xform=wp.transform(self.base_pos),
            floating=True,
            density=1.0,
            armature=1e-4,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e4,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        builder.collapse_fixed_joints()


def object_generator(object_type, **kwargs):
    class __DexObj__(ObjectModel):
        def __init__(self):
            super().__init__(object_type=object_type, **kwargs)

    return __DexObj__


def operable_object_generator(object_type, **kwargs):
    class __OpDexObj__(OperableObjectModel):
        def __init__(self):
            super().__init__(object_type=object_type, **kwargs)

    return __OpDexObj__


StaplerObject = object_generator(ObjectType.TCDM_STAPLER, base_pos=(0.0, 0.01756801, 0.0), scale=1.3)
OctprismObject = object_generator(ObjectType.OCTPRISM, scale=1.0)
SprayBottleObject = operable_object_generator(
    ObjectType.SPRAY_BOTTLE, base_pos=(0.0, 0.22, 0.0), model_path="spray_bottle/mobility.urdf"
)
PillBottleObject = operable_object_generator(
    ObjectType.PILL_BOTTLE,
    base_pos=(0.0, 0.01756801, 0.0),
)

OBJ_MODELS = {}
OBJ_MODELS[ObjectType.TCDM_STAPLER] = StaplerObject
OBJ_MODELS[ObjectType.OCTPRISM] = OctprismObject
OBJ_MODELS[ObjectType.SPRAY_BOTTLE] = SprayBottleObject
OBJ_MODELS[ObjectType.PILL_BOTTLE] = PillBottleObject
