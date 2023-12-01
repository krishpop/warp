# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp

from .integrator_euler import (
    eval_springs,
    eval_triangles,
    eval_triangles_contact,
    eval_bending,
    eval_tetrahedra,
    eval_particle_ground_contacts,
    eval_particle_contacts,
    eval_muscles,
    integrate_particles,
    eval_rigid_contacts,
)


# Frank & Park definition 3.20, pg 100
@wp.func
def spatial_transform_twist(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_transform_wrench(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    v = wp.quat_rotate(q, v)
    w = wp.quat_rotate(q, w) + wp.cross(p, v)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_transform_inverse(t: wp.transform):
    p = wp.transform_get_translation(t)
    q = wp.transform_get_rotation(t)

    q_inv = wp.quat_inverse(q)
    return wp.transform(wp.quat_rotate(q_inv, p) * (0.0 - 1.0), q_inv)


@wp.func
def spatial_adjoint(R: wp.mat33, S: wp.mat33):
    # T = [R  0]
    #     [S  R]

    # fmt: off
    return wp.spatial_matrix(
        R[0, 0], R[0, 1], R[0, 2],     0.0,     0.0,     0.0,
        R[1, 0], R[1, 1], R[1, 2],     0.0,     0.0,     0.0,
        R[2, 0], R[2, 1], R[2, 2],     0.0,     0.0,     0.0,
        S[0, 0], S[0, 1], S[0, 2], R[0, 0], R[0, 1], R[0, 2],
        S[1, 0], S[1, 1], S[1, 2], R[1, 0], R[1, 1], R[1, 2],
        S[2, 0], S[2, 1], S[2, 2], R[2, 0], R[2, 1], R[2, 2],
    )
    # fmt: on


@wp.kernel
def compute_spatial_inertia(
    body_inertia: wp.array(dtype=wp.mat33),
    body_mass: wp.array(dtype=float),
    # outputs
    body_I_m: wp.array(dtype=wp.spatial_matrix),
):
    tid = wp.tid()
    I = body_inertia[tid]
    m = body_mass[tid]
    # fmt: off
    body_I_m[tid] = wp.spatial_matrix(
        I[0, 0], I[0, 1], I[0, 2], 0.0, 0.0, 0.0,
        I[1, 0], I[1, 1], I[1, 2], 0.0, 0.0, 0.0,
        I[2, 0], I[2, 1], I[2, 2], 0.0, 0.0, 0.0,
        0.0,     0.0,     0.0,     m,   0.0, 0.0,
        0.0,     0.0,     0.0,     0.0, m,   0.0,
        0.0,     0.0,     0.0,     0.0, 0.0, m,
    )
    # fmt: on


@wp.kernel
def compute_com_transforms(
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_X_com: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())


# computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
@wp.func
def spatial_transform_inertia(t: wp.transform, I: wp.spatial_matrix):
    t_inv = spatial_transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.mat33(r1, r2, r3)
    S = wp.skew(p) @ R

    T = spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


# compute transform across a joint
@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array(dtype=float),
    start: int,
):
    # prismatic
    if type == wp.sim.JOINT_PRISMATIC:
        q = joint_q[start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    # revolute
    if type == wp.sim.JOINT_REVOLUTE:
        q = joint_q[start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(axis, q))
        return X_jc

    # ball
    if type == wp.sim.JOINT_BALL:
        qx = joint_q[start + 0]
        qy = joint_q[start + 1]
        qz = joint_q[start + 2]
        qw = joint_q[start + 3]

        X_jc = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(qx, qy, qz, qw))
        return X_jc

    # fixed
    if type == wp.sim.JOINT_FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    # free
    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        px = joint_q[start + 0]
        py = joint_q[start + 1]
        pz = joint_q[start + 2]

        qx = joint_q[start + 3]
        qy = joint_q[start + 4]
        qz = joint_q[start + 5]
        qw = joint_q[start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_qd: wp.array(dtype=float),
    qd_start: int,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
):
    # prismatic
    if type == wp.sim.JOINT_PRISMATIC:
        axis = joint_axis[axis_start]
        S_s = spatial_transform_twist(X_sc, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), axis))
        v_j_s = S_s * joint_qd[qd_start]

        wp.store(joint_S_s, qd_start, S_s)
        return v_j_s

    # revolute
    if type == wp.sim.JOINT_REVOLUTE:
        axis = joint_axis[axis_start]
        S_s = spatial_transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3(0.0, 0.0, 0.0)))
        v_j_s = S_s * joint_qd[qd_start]

        wp.store(joint_S_s, qd_start, S_s)
        return v_j_s

    # ball
    if type == wp.sim.JOINT_BALL:
        w = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])

        S_0 = spatial_transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        S_1 = spatial_transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        S_2 = spatial_transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

        # write motion subspace
        wp.store(joint_S_s, qd_start + 0, S_0)
        wp.store(joint_S_s, qd_start + 1, S_1)
        wp.store(joint_S_s, qd_start + 2, S_2)

        return S_0 * w[0] + S_1 * w[1] + S_2 * w[2]

    # fixed
    if type == wp.sim.JOINT_FIXED:
        return wp.spatial_vector()

    # free
    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        v_j_s = wp.spatial_vector(
            joint_qd[qd_start + 0],
            joint_qd[qd_start + 1],
            joint_qd[qd_start + 2],
            joint_qd[qd_start + 3],
            joint_qd[qd_start + 4],
            joint_qd[qd_start + 5],
        )

        # write motion subspace
        wp.store(joint_S_s, qd_start + 0, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        wp.store(joint_S_s, qd_start + 1, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        wp.store(joint_S_s, qd_start + 2, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        wp.store(joint_S_s, qd_start + 3, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        wp.store(joint_S_s, qd_start + 4, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        wp.store(joint_S_s, qd_start + 5, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()


# # compute the velocity across a joint
# #@wp.func
# def jcalc_velocity(self, type, S_s, joint_qd, start):

#     # prismatic
#     if (type == 0):
#         v_j_s = S_s[start)*joint_qd[start]
#         return v_j_s

#     # revolute
#     if (type == 1):
#         v_j_s = S_s[start)*joint_qd[start]
#         return v_j_s

#     # fixed
#     if (type == 2):
#         v_j_s = wp.spatial_vector()
#         return v_j_s

#     # free
#     if (type == 3):
#         v_j_s =  S_s[start+0]*joint_qd[start+0]
#         v_j_s += S_s[start+1]*joint_qd[start+1]
#         v_j_s += S_s[start+2]*joint_qd[start+2]
#         v_j_s += S_s[start+3]*joint_qd[start+3]
#         v_j_s += S_s[start+4]*joint_qd[start+4]
#         v_j_s += S_s[start+5]*joint_qd[start+5]
#         return v_j_s


# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    target_k_e: float,
    target_k_d: float,
    limit_k_e: float,
    limit_k_d: float,
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    body_f_s: wp.spatial_vector,
    # outputs
    tau: wp.array(dtype=float),
):
    # prismatic / revolute
    if type == wp.sim.JOINT_PRISMATIC or type == wp.sim.JOINT_REVOLUTE:
        S_s = joint_S_s[dof_start]

        q = joint_q[coord_start]
        qd = joint_qd[dof_start]
        act = joint_act[dof_start]

        target = joint_target[coord_start]
        lower = joint_limit_lower[coord_start]
        upper = joint_limit_upper[coord_start]

        limit_f = 0.0

        # compute limit forces, damping only active when limit is violated
        if q < lower:
            limit_f = limit_k_e * (lower - q)

        if q > upper:
            limit_f = limit_k_e * (upper - q)

        damping_f = -limit_k_d * qd

        # total torque / force on the joint
        t = -wp.dot(S_s, body_f_s) - target_k_e * (q - target) - target_k_d * qd + act + limit_f + damping_f

        tau[dof_start] = t

    # ball
    if type == wp.sim.JOINT_BALL:
        # elastic term.. this is proportional to the
        # imaginary part of the relative quaternion
        r_j = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # angular velocity for damping
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        for i in range(3):
            S_s = joint_S_s[dof_start + i]

            w = w_j[i]
            r = r_j[i]

            tau[dof_start + i] = -wp.dot(S_s, body_f_s) - w * target_k_d - r * target_k_e

    # fixed
    # if (type == wp.sim.JOINT_FIXED)
    #    pass

    # free
    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = -wp.dot(S_s, body_f_s)


@wp.func
def jcalc_integrate(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    if type == wp.sim.JOINT_FIXED:
        return

    # prismatic / revolute
    if type == wp.sim.JOINT_PRISMATIC or type == wp.sim.JOINT_REVOLUTE:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == wp.sim.JOINT_BALL:
        m_j = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])

        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        w_j_new = w_j + m_j * dt

        drdt_j = wp.quat(w_j_new, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        # update joint vel
        joint_qd_new[dof_start + 0] = w_j_new[0]
        joint_qd_new[dof_start + 1] = w_j_new[1]
        joint_qd_new[dof_start + 2] = w_j_new[2]

        return

    # free joint
    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear acceleration
        m_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])

        a_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        # angular and linear velocity
        w_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        v_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # symplectic Euler
        w_s = w_s + m_s * dt
        v_s = v_s + a_s * dt

        # translation of origin
        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # linear vel of origin (note q/qd switch order of linear angular elements)
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velcity
        dpdt_s = v_s + wp.cross(w_s, p_s)

        # quat and quat derivative
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.quat(w_s, 0.0) * r_s * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        # update transform
        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        # update joint_twist
        joint_qd_new[dof_start + 0] = w_s[0]
        joint_qd_new[dof_start + 1] = w_s[1]
        joint_qd_new[dof_start + 2] = w_s[2]
        joint_qd_new[dof_start + 3] = v_s[0]
        joint_qd_new[dof_start + 4] = v_s[1]
        joint_qd_new[dof_start + 5] = v_s[2]

        return

    # other joint types (compound, universal, D6)
    if type == wp.sim.JOINT_COMPOUND or type == wp.sim.JOINT_UNIVERSAL or type == wp.sim.JOINT_D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qdd = joint_qdd[dof_start + i]
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]

            qd_new = qd + qdd * dt
            q_new = q + qd_new * dt

            joint_qd_new[dof_start + i] = qd_new
            joint_q_new[coord_start + i] = q_new


@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # parent transform
    parent = joint_parent[i]
    child = joint_child[i]

    # parent transform in spatial coordinates
    X_pj = joint_X_p[i]
    X_cj = joint_X_c[i]
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    type = joint_type[i]
    axis_start = joint_axis_start[i]
    lin_axis_count = joint_axis_dim[i, 0]
    ang_axis_count = joint_axis_dim[i, 1]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_j = jcalc_transform(type, joint_axis, axis_start, lin_axis_count, ang_axis_count, joint_q, coord_start)

    # transform from world to joint anchor frame at child body
    X_wcj = X_wpj * X_j
    # transform from world to child body frame
    X_wc = X_wcj * wp.transform_inverse(X_cj)

    # compute transform of center of mass
    X_cm = body_X_com[child]
    # TODO does X_cj influence X_cm like this?
    X_sm = X_wc * X_cm

    # store geometry transforms
    body_q[child] = X_wc
    body_q_com[child] = X_sm


@wp.kernel
def eval_rigid_fk(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_q,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            joint_axis_start,
            joint_axis_dim,
            body_q,
            body_q_com,
        )


@wp.func
def spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_top(a)
    v_a = wp.spatial_bottom(a)

    w_b = wp.spatial_top(b)
    v_b = wp.spatial_bottom(b)

    w = wp.cross(w_a, w_b)
    v = wp.cross(w_a, v_b) + wp.cross(v_a, w_b)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_top(a)
    v_a = wp.spatial_bottom(a)

    w_b = wp.spatial_top(b)
    v_b = wp.spatial_bottom(b)

    w = wp.cross(w_a, w_b) + wp.cross(v_a, v_b)
    v = wp.cross(w_a, v_b)

    return wp.spatial_vector(w, v)


@wp.func
def dense_index(stride: int, i: int, j: int):
    return i * stride + j


@wp.func
def compute_link_velocity(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    type = joint_type[i]
    axis = joint_axis[i]
    child = joint_child[i]
    parent = joint_parent[i]
    qd_start = joint_qd_start[i]

    # parent transform in spatial coordinates
    X_sp = wp.transform_identity()
    if parent >= 0:
        X_sp = body_q[parent]

    X_pj = joint_X_p[i]
    X_sj = X_sp * X_pj

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    axis_start = joint_axis_start[i]
    lin_axis_count = joint_axis_dim[i, 0]
    ang_axis_count = joint_axis_dim[i, 1]
    v_j_s = jcalc_motion(type, joint_axis, axis_start, lin_axis_count, ang_axis_count, X_sj, joint_qd, qd_start, joint_S_s)

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if parent >= 0:
        v_parent_s = body_v_s[parent]
        a_parent_s = body_a_s[parent]

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s)  # + self.joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = body_q_com[child]
    I_m = body_I_m[child]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    m = I_m[3, 3]

    f_g_m = wp.spatial_vector(wp.vec3(), gravity) * m
    f_g_s = spatial_transform_wrench(wp.transform(wp.transform_get_translation(X_sm), wp.quat_identity()), f_g_m)

    # f_ext_s = body_f_s[i] + f_g_s

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = wp.mul(I_s, a_s) + spatial_cross_dual(v_s, wp.mul(I_s, v_s))

    wp.store(body_v_s, child, v_s)
    wp.store(body_a_s, child, a_s)
    wp.store(body_f_s, child, f_b_s - f_g_s)
    wp.store(body_I_s, child, I_s)


@wp.kernel
def eval_rigid_id(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_qd,
            joint_axis,
            joint_axis_start,
            joint_axis_dim,
            body_I_m,
            body_q,
            body_q_com,
            joint_X_p,
            gravity,
            joint_S_s,
            body_I_s,
            body_v_s,
            body_f_s,
            body_a_s,
        )


@wp.func
def compute_link_tau(
    offset: int,
    joint_end: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # for backwards traversal
    i = joint_end - offset - 1

    type = joint_type[i]
    parent = joint_parent[i]
    child = joint_child[i]
    dof_start = joint_qd_start[i]
    coord_start = joint_q_start[i]

    target_k_e = joint_target_ke[i]
    target_k_d = joint_target_kd[i]

    limit_k_e = joint_limit_ke[i]
    limit_k_d = joint_limit_kd[i]

    # total forces on body
    f_b_s = body_fb_s[child]
    f_t_s = body_ft_s[child]

    f_s = f_b_s + f_t_s

    # compute joint-space forces, writes out tau
    jcalc_tau(
        type,
        target_k_e,
        target_k_d,
        limit_k_e,
        limit_k_d,
        joint_S_s,
        joint_q,
        joint_qd,
        joint_act,
        joint_target,
        joint_limit_lower,
        joint_limit_upper,
        coord_start,
        dof_start,
        f_s,
        tau,
    )

    # update parent forces, todo: check that this is valid for the backwards pass
    if parent >= 0:
        wp.atomic_add(body_ft_s, parent, f_s)


@wp.kernel
def eval_rigid_tau(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]
    count = end - start

    # compute joint forces
    for i in range(count):
        compute_link_tau(
            i,
            end,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_qd,
            joint_act,
            joint_target,
            joint_target_ke,
            joint_target_kd,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_S_s,
            body_fb_s,
            body_ft_s,
            tau,
        )


# builds spatial Jacobian J which is an (joint_count*6)x(dof_count) matrix
@wp.func
def spatial_jacobian(
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_parent: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_start: int,  # offset of the first joint for the articulation
    joint_count: int,
    J_start: int,
    # outputs
    J: wp.array(dtype=float),
):
    articulation_dof_start = joint_qd_start[joint_start]
    articulation_dof_end = joint_qd_start[joint_start + joint_count]
    articulation_dof_count = articulation_dof_end - articulation_dof_start

    # # shift output pointers
    # const int S_start = articulation_dof_start;

    # S += S_start;
    # J += J_start;

    for i in range(joint_count):
        row_start = (articulation_dof_start + i) * 6

        j = joint_start + i
        while j != -1:
            joint_dof_start = joint_qd_start[j]
            joint_dof_end = joint_qd_start[j + 1]
            joint_dof_count = joint_dof_end - joint_dof_start

            # fill out each row of the Jacobian walking up the tree
            for dof in range(joint_dof_count):
                col = (joint_dof_start - articulation_dof_start) + dof
                S = joint_S_s[col + articulation_dof_start]

                for k in range(6):
                    J[J_start + dense_index(articulation_dof_count, row_start + k, col)] = S[k]

                # J[row_start+0, col] = S.w.x
                # J[row_start+1, col] = S.w.y
                # J[row_start+2, col] = S.w.z
                # J[row_start+3, col] = S.v.x
                # J[row_start+4, col] = S.v.y
                # J[row_start+5, col] = S.v.z

            j = joint_parent[j]


@wp.kernel
def eval_rigid_jacobian(
    articulation_start: wp.array(dtype=int),
    articulation_J_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    J: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    joint_start = articulation_start[index]
    joint_end = articulation_start[index + 1]
    joint_count = joint_end - joint_start

    J_offset = articulation_J_start[index]

    spatial_jacobian(joint_S_s, joint_parent, joint_qd_start, joint_start, joint_count, J_offset, J)


@wp.func
def spatial_mass(
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    joint_start: int,
    joint_count: int,
    M_start: int,
    # outputs
    M: wp.array(dtype=float),
):
    stride = joint_count * 6

    for l in range(joint_count):
        I = body_I_s[joint_start + l]
        for i in range(6):
            for j in range(6):
                M[M_start + dense_index(stride, l * 6 + i, l * 6 + j)] = I[i, j]


@wp.kernel
def eval_rigid_mass(
    articulation_start: wp.array(dtype=int),
    articulation_M_start: wp.array(dtype=int),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    # outputs
    M: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    joint_start = articulation_start[index]
    joint_end = articulation_start[index + 1]
    joint_count = joint_end - joint_start

    M_offset = articulation_M_start[index]

    spatial_mass(body_I_s, joint_start, joint_count, M_offset, M)


@wp.func
def dense_gemm(
    m: int,
    n: int,
    p: int,
    transpose_A: bool,
    transpose_B: bool,
    add_to_C: bool,
    A_start: int,
    B_start: int,
    C_start: int,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    # outputs
    C: wp.array(dtype=float),
):
    # multiply a `m x p` matrix A by a `p x n` matrix B to produce a `m x n` matrix C
    for i in range(m):
        for j in range(n):
            sum = float(0.0)
            for k in range(p):
                if transpose_A:
                    a_i = k * m + i
                else:
                    a_i = i * p + k
                if transpose_B:
                    b_j = j * p + k
                else:
                    b_j = k * n + j
                sum += A[A_start + a_i] * B[B_start + b_j]

            if add_to_C:
                C[C_start + i * n + j] += sum
            else:
                C[C_start + i * n + j] = sum


@wp.kernel
def eval_dense_gemm_batched(
    m: wp.array(dtype=int),
    n: wp.array(dtype=int),
    p: wp.array(dtype=int),
    transpose_A: bool,
    transpose_B: bool,
    A_start: wp.array(dtype=int),
    B_start: wp.array(dtype=int),
    C_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    C: wp.array(dtype=float),
):
    # on the CPU each thread computes the whole matrix multiply
    # on the GPU each block computes the multiply with one output per-thread
    batch = wp.tid()  # /kNumThreadsPerBlock;
    add_to_C = False

    dense_gemm(
        m[batch],
        n[batch],
        p[batch],
        transpose_A,
        transpose_B,
        add_to_C,
        A_start[batch],
        B_start[batch],
        C_start[batch],
        A,
        B,
        C,
    )


@wp.func
def dense_cholesky(
    n: int,
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array(dtype=float),
):
    # compute the Cholesky factorization of A = L L^T with diagonal regularization R
    for j in range(n):
        s = A[A_start + dense_index(n, j, j)] + R[R_start + j]

        for k in range(j):
            r = L[A_start + dense_index(n, j, k)]
            s -= r * r

        s = wp.sqrt(s)
        invS = 1.0 / s

        L[A_start + dense_index(n, j, j)] = s

        for i in range(j + 1, n):
            s = A[A_start + dense_index(n, i, j)]

            for k in range(j):
                s -= L[A_start + dense_index(n, i, k)] * L[A_start + dense_index(n, j, k)]

            L[A_start + dense_index(n, i, j)] = s * invS


@wp.kernel
def eval_dense_cholesky_batched(
    A_starts: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    L: wp.array(dtype=float),
):
    batch = wp.tid()

    n = A_dim[batch]
    A_start = A_starts[batch]
    R_start = n * batch

    dense_cholesky(n, A, R, A_start, R_start, L)


@wp.func
def dense_subs(
    n: int,
    L_start: int,
    b_start: int,
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
):
    # Solves (L L^T) x = b for x given the Cholesky factor L
    # forward substitution solves the lower triangular system L y = b for y
    for i in range(n):
        s = b[b_start + i]

        for j in range(i):
            s -= L[L_start + dense_index(n, i, j)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]

    # backward substitution solves the upper triangular system L^T x = y for x
    for i in range(n - 1, -1, -1):
        s = x[b_start + i]

        for j in range(i + 1, n):
            s -= L[L_start + dense_index(n, j, i)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]


@wp.kernel
def eval_dense_solve_batched(
    L_start: wp.array(dtype=int),
    L_dim: wp.array(dtype=int),
    b_start: wp.array(dtype=int),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    x: wp.array(dtype=float),
):
    batch = wp.tid()

    dense_subs(L_dim[batch], L_start[batch], b_start[batch], L, b, x)


@wp.kernel
def integrate_generalized_joints(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    type = joint_type[index]
    coord_start = joint_q_start[index]
    dof_start = joint_qd_start[index]
    lin_axis_count = joint_axis_dim[index, 0]
    ang_axis_count = joint_axis_dim[index, 1]

    jcalc_integrate(
        type,
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        lin_axis_count,
        ang_axis_count,
        dt,
        joint_q_new,
        joint_qd_new,
    )


class FeatherstoneIntegrator:
    """A semi-implicit integrator using symplectic Euler

    After constructing `Model` and `State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example
    -------

    .. code-block:: python

        integrator = wp.SemiImplicitIntegrator()

        # simulation loop
        for i in range(100):
            state = integrator.simulate(model, state_in, state_out, dt)

    """

    def __init__(self, angular_damping=0.05, plugins=[]):
        self.angular_damping = angular_damping
        self.plugins = plugins

    def register_plugin(self, plugin):
        self.plugins.append(plugin)

    def augment_state(self, model, state):
        if model.body_count:
            # joints
            state.joint_q = wp.clone(model.joint_q, requires_grad=state.requires_grad)
            state.joint_qd = wp.clone(model.joint_qd, requires_grad=state.requires_grad)
            state.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=state.requires_grad)
            state.joint_tau = wp.empty_like(model.joint_qd, requires_grad=state.requires_grad)
            state.joint_S_s = wp.empty(
                (model.joint_dof_count,),
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=state.requires_grad,
            )

            # derived rigid body data (maximal coordinates)
            state.body_q_com = wp.empty_like(state.body_q, requires_grad=state.requires_grad)
            state.body_I_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=state.requires_grad
            )
            state.body_v_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=state.requires_grad
            )
            state.body_a_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=state.requires_grad
            )
            state.body_f_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=state.requires_grad
            )
            state.body_ft_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=state.requires_grad
            )
            state.body_f_ext_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=state.requires_grad
            )

        for plugin in self.plugins:
            if not plugin.initialized:
                plugin.initialize(model, self)
            plugin.augment_state(model, state)

    def augment_model(self, model):
        # allocate mass matrix
        if model.joint_count:
            # calculate total size and offsets of Jacobian and mass matrices for entire system
            model.J_size = 0
            model.M_size = 0
            model.H_size = 0

            articulation_J_start = []
            articulation_M_start = []
            articulation_H_start = []

            articulation_M_rows = []
            articulation_H_rows = []
            articulation_J_rows = []
            articulation_J_cols = []

            articulation_dof_start = []
            articulation_coord_start = []

            articulation_start = model.articulation_start.numpy()
            joint_q_start = model.joint_q_start.numpy()
            joint_qd_start = model.joint_qd_start.numpy()

            for i in range(model.articulation_count):
                first_joint = articulation_start[i]
                last_joint = articulation_start[i + 1]

                first_coord = joint_q_start[first_joint]

                first_dof = joint_qd_start[first_joint]
                last_dof = joint_qd_start[last_joint]

                joint_count = last_joint - first_joint
                dof_count = last_dof - first_dof

                articulation_J_start.append(model.J_size)
                articulation_M_start.append(model.M_size)
                articulation_H_start.append(model.H_size)
                articulation_dof_start.append(first_dof)
                articulation_coord_start.append(first_coord)

                # bit of data duplication here, but will leave it as such for clarity
                articulation_M_rows.append(joint_count * 6)
                articulation_H_rows.append(dof_count)
                articulation_J_rows.append(joint_count * 6)
                articulation_J_cols.append(dof_count)

                model.J_size += 6 * joint_count * dof_count
                model.M_size += 6 * joint_count * 6 * joint_count
                model.H_size += dof_count * dof_count

            # matrix offsets for batched gemm
            model.articulation_J_start = wp.array(articulation_J_start, dtype=wp.int32, device=model.device)
            model.articulation_M_start = wp.array(articulation_M_start, dtype=wp.int32, device=model.device)
            model.articulation_H_start = wp.array(articulation_H_start, dtype=wp.int32, device=model.device)

            model.articulation_M_rows = wp.array(articulation_M_rows, dtype=wp.int32, device=model.device)
            model.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32, device=model.device)
            model.articulation_J_rows = wp.array(articulation_J_rows, dtype=wp.int32, device=model.device)
            model.articulation_J_cols = wp.array(articulation_J_cols, dtype=wp.int32, device=model.device)

            model.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32, device=model.device)
            model.articulation_coord_start = wp.array(articulation_coord_start, dtype=wp.int32, device=model.device)

            # system matrices
            model.M = wp.zeros(
                (model.M_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad
            )
            model.J = wp.zeros(
                (model.J_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad
            )
            model.P = wp.empty_like(model.J, requires_grad=model.requires_grad)
            model.H = wp.empty(
                (model.H_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad
            )

            # zero since only upper triangle is set which can trigger NaN detection
            model.L = wp.zeros_like(model.H, requires_grad=model.requires_grad)

        if model.body_count:
            model.body_I_m = wp.empty((model.body_count,), dtype=wp.spatial_matrix, device=model.device)
            wp.launch(
                compute_spatial_inertia,
                model.body_count,
                inputs=[model.body_inertia, model.body_mass],
                outputs=[model.body_I_m],
                device=model.device,
            )
            model.body_X_com = wp.empty((model.body_count,), dtype=wp.transform, device=model.device)
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[model.body_X_com],
                device=model.device,
            )

    def simulate(self, model, state_in, state_out, dt, update_mass_matrix=True):
        requires_grad = state_in.requires_grad
        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            # damped springs
            if model.spring_count:
                wp.launch(
                    eval_springs,
                    dim=model.spring_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.spring_indices,
                        model.spring_rest_length,
                        model.spring_stiffness,
                        model.spring_damping,
                    ],
                    outputs=[state_out.particle_f],
                    device=model.device,
                )

            # triangle elastic and lift/drag forces
            if model.tri_count and model.tri_ke > 0.0:
                wp.launch(
                    eval_triangles,
                    dim=model.tri_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_activations,
                        model.tri_ke,
                        model.tri_ka,
                        model.tri_kd,
                        model.tri_drag,
                        model.tri_lift,
                    ],
                    outputs=[state_out.particle_f],
                    device=model.device,
                )

            # triangle/triangle contacts
            if model.enable_tri_collisions and model.tri_count and model.tri_ke > 0.0:
                wp.launch(
                    eval_triangles_contact,
                    dim=model.tri_count * model.particle_count,
                    inputs=[
                        model.particle_count,
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_activations,
                        model.tri_ke,
                        model.tri_ka,
                        model.tri_kd,
                        model.tri_drag,
                        model.tri_lift,
                    ],
                    outputs=[state_out.particle_f],
                    device=model.device,
                )

            # triangle bending
            if model.edge_count:
                wp.launch(
                    eval_bending,
                    dim=model.edge_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.edge_indices,
                        model.edge_rest_angle,
                        model.edge_ke,
                        model.edge_kd,
                    ],
                    outputs=[state_out.particle_f],
                    device=model.device,
                )

            # particle shape contact
            if model.particle_count and model.shape_count > 1:
                if state_in.has_soft_contact_vars:
                    contact_state = state_in
                else:
                    contact_state = model
                wp.launch(
                    kernel=eval_particle_contacts,
                    dim=model.soft_contact_max,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        state_in.body_q,
                        state_in.body_qd,
                        model.particle_radius,
                        model.particle_flags,
                        model.body_com,
                        model.shape_body,
                        model.shape_materials,
                        model.soft_contact_ke,
                        model.soft_contact_kd,
                        model.soft_contact_kf,
                        model.soft_contact_mu,
                        model.particle_adhesion,
                        contact_state.soft_contact_count,
                        contact_state.soft_contact_particle,
                        contact_state.soft_contact_shape,
                        contact_state.soft_contact_body_pos,
                        contact_state.soft_contact_body_vel,
                        contact_state.soft_contact_normal,
                        model.soft_contact_max,
                    ],
                    # outputs
                    outputs=[particle_f, body_f],
                    device=model.device,
                )

            # tetrahedral FEM
            if model.tet_count:
                wp.launch(
                    eval_tetrahedra,
                    dim=model.tet_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.tet_indices,
                        model.tet_poses,
                        model.tet_activations,
                        model.tet_materials,
                    ],
                    outputs=[state_out.particle_f],
                    device=model.device,
                )

            # ----------------------------
            # articulations

            if model.body_count:
                # evaluate body transforms
                wp.launch(
                    eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        state_in.joint_q,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.body_X_com,
                        model.joint_axis,
                        model.joint_axis_start,
                        model.joint_axis_dim,
                    ],
                    outputs=[state_in.body_q, state_in.body_q_com],
                    device=model.device,
                )

                # evaluate joint inertias, motion vectors, and forces
                wp.launch(
                    eval_rigid_id,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_qd_start,
                        state_in.joint_qd,
                        model.joint_axis,
                        model.joint_axis_start,
                        model.joint_axis_dim,
                        model.body_I_m,
                        state_in.body_q,
                        state_in.body_q_com,
                        model.joint_X_p,
                        model.gravity,
                    ],
                    outputs=[
                        state_out.joint_S_s,
                        state_out.body_I_s,
                        state_out.body_v_s,
                        state_out.body_f_s,
                        state_out.body_a_s,
                    ],
                    device=model.device,
                )

                if model.rigid_contact_max and (
                    model.ground and model.shape_ground_contact_pair_count or model.shape_contact_pair_count
                ):
                    if state_in.has_rigid_contact_vars:
                        contact_state = state_in
                    else:
                        contact_state = model
                    wp.launch(
                        kernel=eval_rigid_contacts,
                        dim=model.rigid_contact_max,
                        inputs=[
                            state_in.body_q,
                            state_in.body_qd,
                            model.body_com,
                            model.shape_materials,
                            model.shape_geo,
                            model.shape_body,
                            contact_state.rigid_contact_count,
                            contact_state.rigid_contact_point0,
                            contact_state.rigid_contact_point1,
                            contact_state.rigid_contact_normal,
                            contact_state.rigid_contact_shape0,
                            contact_state.rigid_contact_shape1,
                        ],
                        outputs=[body_f],
                        device=model.device,
                    )

                # particle shape contact
                if model.particle_count and model.shape_count > 1:
                    if state_in.has_soft_contact_vars:
                        contact_state = state_in
                    else:
                        contact_state = model
                    wp.launch(
                        kernel=eval_particle_contacts,
                        dim=model.soft_contact_max,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            state_in.body_q,
                            state_in.body_qd,
                            model.particle_radius,
                            model.particle_flags,
                            model.body_com,
                            model.shape_body,
                            model.shape_materials,
                            model.soft_contact_ke,
                            model.soft_contact_kd,
                            model.soft_contact_kf,
                            model.soft_contact_mu,
                            model.particle_adhesion,
                            contact_state.soft_contact_count,
                            contact_state.soft_contact_particle,
                            contact_state.soft_contact_shape,
                            contact_state.soft_contact_body_pos,
                            contact_state.soft_contact_body_vel,
                            contact_state.soft_contact_normal,
                            model.soft_contact_max,
                        ],
                        outputs=[particle_f, body_f],
                        device=model.device,
                    )

                # evaluate muscle actuation
                if False and model.muscle_count:
                    wp.launch(
                        eval_muscles,
                        dim=model.muscle_count,
                        inputs=[
                            state_out.body_q,
                            state_out.body_v_s,
                            model.muscle_start,
                            model.muscle_params,
                            model.muscle_links,
                            model.muscle_points,
                            model.muscle_activation,
                        ],
                        outputs=[state_out.body_f_s],
                        device=model.device,
                    )

                # evaluate joint torques
                state_out.body_ft_s.zero_()
                wp.launch(
                    eval_rigid_tau,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        state_in.joint_qd,
                        state_in.joint_act,
                        model.joint_target,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_limit_ke,
                        model.joint_limit_kd,
                        state_out.joint_S_s,
                        state_out.body_f_s,
                    ],
                    outputs=[
                        state_out.body_ft_s,
                        state_out.joint_tau,
                    ],
                    device=model.device,
                )

                if update_mass_matrix:

                    # build J
                    wp.launch(
                        eval_rigid_jacobian,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.articulation_J_start,
                            model.joint_parent,
                            model.joint_qd_start,
                            state_out.joint_S_s,
                        ],
                        outputs=[model.J],
                        device=model.device,
                    )

                    # build M
                    wp.launch(
                        eval_rigid_mass,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.articulation_M_start,
                            state_out.body_I_s,
                        ],
                        outputs=[model.M],
                        device=model.device,
                    )

                    # form P = M*J
                    wp.launch(
                        eval_dense_gemm_batched,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_M_rows,
                            model.articulation_J_cols,
                            model.articulation_J_rows,
                            False,
                            False,
                            model.articulation_M_start,
                            model.articulation_J_start,
                            # P start is the same as J start since it has the same dims as J
                            model.articulation_J_start,
                            model.M,
                            model.J,
                        ],
                        outputs=[model.P],
                        device=model.device,
                    )

                    # form H = J^T*P
                    wp.launch(
                        eval_dense_gemm_batched,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_J_cols,
                            model.articulation_J_cols,
                            # P rows is the same as J rows
                            model.articulation_J_rows,
                            True,
                            False,
                            model.articulation_J_start,
                            # P start is the same as J start since it has the same dims as J
                            model.articulation_J_start,
                            model.articulation_H_start,
                            model.J,
                            model.P,
                        ],
                        outputs=[model.H],
                        device=model.device,
                    )

                    # compute decomposition
                    wp.launch(
                        eval_dense_cholesky_batched,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_H_start,
                            model.articulation_H_rows,
                            model.H,
                            model.joint_armature,
                        ],
                        outputs=[model.L],
                        device=model.device,
                    )

                # solve for qdd
                wp.launch(
                    eval_dense_solve_batched,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_H_start,
                        model.articulation_H_rows,
                        model.articulation_dof_start,
                        model.L,
                        state_out.joint_tau,
                    ],
                    outputs=[state_out.joint_qdd],
                    device=model.device,
                )

            for plugin in self.plugins:
                plugin.before_integrate(model, state_in, state_out, dt, requires_grad)

            # -------------------------------------
            # integrate bodies

            if model.joint_count:
                wp.launch(
                    kernel=integrate_generalized_joints,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_axis_dim,
                        state_in.joint_q,
                        state_in.joint_qd,
                        state_out.joint_qdd,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    device=model.device,
                )

                wp.launch(
                    eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        state_out.joint_q,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.body_X_com,
                        model.joint_axis,
                        model.joint_axis_start,
                        model.joint_axis_dim,
                    ],
                    outputs=[state_out.body_q, state_out.body_q_com],
                    device=model.device,
                )

            # ----------------------------
            # integrate particles

            if model.particle_count:
                wp.launch(
                    kernel=integrate_particles,
                    dim=model.particle_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        state_in.particle_f,
                        model.particle_inv_mass,
                        model.particle_flags,
                        model.gravity,
                        dt,
                        model.particle_max_velocity,
                    ],
                    outputs=[state_out.particle_q, state_out.particle_qd],
                    device=model.device,
                )

            for plugin in self.plugins:
                plugin.after_integrate(model, state_in, state_out, dt, requires_grad)

            return state_out
