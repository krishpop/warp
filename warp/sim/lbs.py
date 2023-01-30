# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from warp.sim.model import JOINT_COMPOUND, JOINT_REVOLUTE, JOINT_UNIVERSAL
from warp.utils import transform_identity

import math
import numpy as np
import os

import xml.etree.ElementTree as ET

import warp as wp
import warp.sim 

from warp.utils import quat_to_matrix, quat_rotate, quat_inverse, quat_from_matrix

from .import_mjcf import parse_mjcf

@wp.kernel
def wpk_update_vertices(
            new_positions: wp.array(dtype=wp.vec3),
            old_positions: wp.array(dtype=wp.vec3)
        ):
    
    tid = wp.tid()
    old_positions[tid] = new_positions[tid]

@wp.kernel
def wpk_get_vertices(
            mesh_id: warp.types.uint64, 
            positions: wp.array(dtype=wp.vec3)
        ):
    tid = wp.tid()
    pos = wp.mesh_get_point(mesh_id, tid)
    positions[tid] = pos


def lbs_compute_mass(link, lbs_weights, density=1.0):
    m = lbs_weights[:,link].sum() / lbs_weights.shape[0] * density
    Ia = m 
    I = np.array([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])
    return m, I

def parse_lbs(
    filename, 
    builder, 
    get_lbs_variables,
    lbs_link_id=0,
    density=1000.0, 
    stiffness=0.0, 
    damping=0.0, 
    contact_ke=1000.0,
    contact_kd=100.0,
    contact_kf=100.0,
    contact_mu=0.5,
    contact_restitution=0.5,
    limit_ke=10000.0,
    limit_kd=1000.0,
    armature=0.0,
    armature_scale=1.0,
    parse_meshes=False,
    enable_self_collisions=True):

    lbs_weights, lbs_G, lbs_rest, lbs_base_transform, lbs_scale, lbs_verts, lbs_faces, joint_0_offset = get_lbs_variables()
    # builder.add_lbs(lbs_weights, lbs_G, lbs_rest, lbs_base_transform, lbs_scale, lbs_verts, lbs_faces)    
    # lbs_mesh = wp.sim.Mesh(lbs_verts * lbs_scale, lbs_faces.flatten())
    lbs_mesh = wp.sim.Mesh(lbs_verts, lbs_faces.flatten())

    # lbs_rest = np.hstack((lbs_verts, np.ones((len(lbs_verts), 1))))

    shape_start = builder.shape_count

    parse_mjcf(
        filename,
        builder,
        density,
        stiffness,
        damping,
        contact_ke,
        contact_kd,
        contact_kf,
        contact_mu,
        contact_restitution,
        limit_ke,
        limit_kd,
        armature,
        armature_scale,
        parse_meshes,
        enable_self_collisions)

    # builder.joint_X_p[0] = wp.transform(joint_0_offset, wp.quat_identity())
    builder.joint_X_p[0] = lbs_base_transform
        
    shape_id = builder.add_shape_lbs(
        lbs_link_id,
        lbs_weights, lbs_G, lbs_rest,
        lbs_faces, lbs_verts,
        mesh=lbs_mesh,
        pos=lbs_base_transform.p,
        rot=lbs_base_transform.q,
        scale=(lbs_scale, lbs_scale, lbs_scale),
        ke=contact_ke,
        kd=contact_kd,
        kf=contact_kf,  
        mu=contact_mu,
        density=0,
        restitution=contact_restitution,
    )

    # make sure the surrounding LBS mesh doesn't collide with the geoms
    # added previously (regardless of self-collision setting)
    for i in range(shape_start, builder.shape_count):
        if i != shape_id:
            builder.shape_collision_filter_pairs.add((i, shape_id))

    m, I = lbs_compute_mass(lbs_link_id, lbs_weights)
    builder._update_body_mass(
        lbs_link_id, m, I, np.array([0.,0.,0.]), np.array([0.0, 0.0, 0.0, 1.0]))


def transform_to_matrix(q, scale=1):
    mat = np.eye(4)
    mat[:3, 3] = q[:3] / scale
    mat[:3,:3] = quat_to_matrix(q[3:])
    return mat

def transform_from_matrix(mat, scale=1):
    q = np.zeros([7])
    q[:3] = mat[:3, 3] * scale
    q[3:] = quat_from_matrix(mat[:3,:3])
    return q


def update_state_from_transform(model, joint_transform, obj_transform, is_update_object, state):
    if state is None:
        body_q = model.body_q.numpy()
    else:
        body_q = state.body_q.numpy()
    np_joint_transform = joint_transform[:model.lbs_link_count]
    shape_geo_scale = model.shape_geo_scale.numpy()
    for i in range(len(np_joint_transform)):
        scale = 1.0
        if i in model.lbs_body_ids:
            # retrieve LBS scaling
            scale = shape_geo_scale[model.body_shapes[i][-1]]
        body_q[i] = transform_from_matrix(np_joint_transform[i], scale) 
    if is_update_object and len(body_q) > len(np_joint_transform):
        body_q[len(np_joint_transform)] = transform_from_matrix(obj_transform) 

    model.body_q = wp.array(body_q, dtype=wp.transform)
    prev_q = body_q
    state = model.state()
    wp.sim.eval_ik(model, state, model.joint_q, model.joint_qd)
    wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state)

    curr_q = state.body_q.numpy()
    model.body_q = wp.array(state.body_q.numpy(), dtype=wp.transform)


def update_lbs(model, body_q):
    # Equation 7 from Loper et al., "SMPL: A Skinned Multi-Person Linear Model", 2015

    # TODO implement this in Warp
    import torch

    shape_geo_scale = model.shape_geo_scale.numpy()
    shape_transform = model.shape_transform.numpy()
    # TODO support multiple LBS bodies
    lbs_body_id = next(iter(model.lbs_body_ids))
    lbs_shape_id = model.body_shapes[lbs_body_id][-1]
    lbs_scale = shape_geo_scale[lbs_shape_id][0]
    lbs_base_transform = wp.transform(shape_transform[lbs_shape_id][:3], shape_transform[lbs_shape_id][3:])

    trans = np.zeros(7)
    # trans[:3] = lbs_base_transform.p
    joint_transform = np.array([transform_to_matrix(q - trans, lbs_scale) for q in body_q.numpy()[:model.lbs_link_count]])
    joint_transform = torch.tensor(joint_transform).float()

    
    # reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
    # joint_transform = joint_transform[reorder_idxs]

    # print("joint_transform from body_q:")
    # print(joint_transform)

    G = torch.tensor(model.lbs_G.numpy())
    rest_shape = torch.tensor(model.lbs_rest_shape.numpy())
    lbs_weights = torch.tensor(model.lbs_weights.numpy())

    J = torch.matmul(joint_transform, G).permute(1, 2, 0)
    th_T = torch.matmul(J, lbs_weights.transpose(1, 0)).permute(2, 0, 1)
    th_verts = (th_T * rest_shape.unsqueeze(1)).sum(2)
    th_verts = th_verts[:, :3]

    # th_jtr = joint_transform[:, :3, 3]
    # center_joint = th_jtr[0].unsqueeze(0)
    # th_verts = th_verts - center_joint

    th_verts = th_verts.cpu().detach().numpy() * lbs_scale #+ np.expand_dims(lbs_base_transform.p, 0)
    # th_verts[:,1:] = -th_verts[:,1:]

    # model.lbs_verts = wp.array(th_verts, dtype=wp.vec3)
    
    mesh = model.shape_geo_src[lbs_shape_id].mesh

    # update vertices used in the renderer
    model.shape_geo_src[lbs_shape_id].vertices = th_verts
    # model.shape_geo_src[lbs_shape_id].vertices = rest_shape[:,:3].cpu().detach().numpy()
    # model.shape_geo_src[lbs_shape_id].vertices = model.lbs_verts.numpy()
    # model.shape_geo_src[lbs_shape_id].indices = model.lbs_faces.numpy().flatten()

    # wp.launch(
    #     kernel=wpk_update_vertices,
    #     dim=len(mesh.points),
    #     inputs=[wp.array(th_verts, dtype=wp.vec3), mesh.points])

