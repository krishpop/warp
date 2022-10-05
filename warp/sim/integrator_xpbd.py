# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
from .model import ShapeContactMaterial
from .utils import velocity_at_point
from .integrator_euler import integrate_bodies


@wp.kernel
def integrate_particles(x: wp.array(dtype=wp.vec3),
                        v: wp.array(dtype=wp.vec3),
                        f: wp.array(dtype=wp.vec3),
                        w: wp.array(dtype=float),
                        gravity: wp.vec3,
                        dt: float,
                        x_new: wp.array(dtype=wp.vec3),
                        v_new: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]
    inv_mass = w[tid]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt
    x1 = x0 + v1 * dt

    x_new[tid] = x1
    v_new[tid] = v1

@wp.kernel
def solve_springs(x: wp.array(dtype=wp.vec3),
                  v: wp.array(dtype=wp.vec3),
                  invmass: wp.array(dtype=float),
                  spring_indices: wp.array(dtype=int),
                  spring_rest_lengths: wp.array(dtype=float),
                  spring_stiffness: wp.array(dtype=float),
                  spring_damping: wp.array(dtype=float),
                  dt: float,
                  delta: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = wp.dot(dir, vij)

    # damping based on relative velocity.
    #fs = dir * (ke * c + kd * dcdt)

    wi = invmass[i]
    wj = invmass[j]

    denom = wi + wj
    alpha = 1.0/(ke*dt*dt)

    multiplier = c / (denom)  # + alpha)

    xd = dir*multiplier

    wp.atomic_sub(delta, i, xd*wi)
    wp.atomic_add(delta, j, xd*wj)


@wp.kernel
def solve_tetrahedra(x: wp.array(dtype=wp.vec3),
                     v: wp.array(dtype=wp.vec3),
                     inv_mass: wp.array(dtype=float),
                     indices: wp.array(dtype=int),
                     pose: wp.array(dtype=wp.mat33),
                     activation: wp.array(dtype=float),
                     materials: wp.array(dtype=float),
                     dt: float,
                     relaxation: float,
                     delta: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    i = indices[tid * 4 + 0]
    j = indices[tid * 4 + 1]
    k = indices[tid * 4 + 2]
    l = indices[tid * 4 + 3]

    act = activation[tid]

    k_mu = materials[tid * 3 + 0]
    k_lambda = materials[tid * 3 + 1]
    k_damp = materials[tid * 3 + 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    v0 = v[i]
    v1 = v[j]
    v2 = v[k]
    v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = wp.mat33(x10, x20, x30)
    Dm = pose[tid]

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # # C_sqrt
    # tr = dot(f1, f1) + dot(f2, f2) + dot(f3, f3)
    # r_s = wp.sqrt(abs(tr - 3.0))
    # C = r_s

    # if (r_s == 0.0):
    #     return

    # if (tr < 3.0):
    #     r_s = 0.0 - r_s

    # dCdx = F*wp.transpose(Dm)*(1.0/r_s)
    # alpha = 1.0 + k_mu / k_lambda

    # C_Neo
    r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    r_s_inv = 1.0/r_s
    C = r_s
    dCdx = F*wp.transpose(Dm)*r_s_inv
    alpha = 1.0 + k_mu / k_lambda

    # C_Spherical
    # r_s = wp.sqrt(dot(f1, f1) + dot(f2, f2) + dot(f3, f3))
    # r_s_inv = 1.0/r_s
    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv
    # alpha = 1.0

    # C_D
    #r_s = wp.sqrt(dot(f1, f1) + dot(f2, f2) + dot(f3, f3))
    #C = r_s*r_s - 3.0
    #dCdx = F*wp.transpose(Dm)*2.0
    #alpha = 1.0

    grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    denom = wp.dot(grad0, grad0)*w0 + wp.dot(grad1, grad1)*w1 + \
        wp.dot(grad2, grad2)*w2 + wp.dot(grad3, grad3)*w3
    multiplier = C/(denom + 1.0/(k_mu*dt*dt*rest_volume))

    delta0 = grad0*multiplier
    delta1 = grad1*multiplier
    delta2 = grad2*multiplier
    delta3 = grad3*multiplier

    # hydrostatic part
    J = wp.determinant(F)

    C_vol = J - alpha
    # dCdx = wp.mat33(cross(f2, f3), cross(f3, f1), cross(f1, f2))*wp.transpose(Dm)

    # grad1 = float3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = float3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = float3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    s = inv_rest_volume / 6.0
    grad1 = wp.cross(x20, x30) * s
    grad2 = wp.cross(x30, x10) * s
    grad3 = wp.cross(x10, x20) * s
    grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    denom = wp.dot(grad0, grad0)*w0 + wp.dot(grad1, grad1)*w1 + \
        wp.dot(grad2, grad2)*w2 + wp.dot(grad3, grad3)*w3
    multiplier = C_vol/(denom + 1.0/(k_lambda*dt*dt*rest_volume))

    delta0 = delta0 + grad0 * multiplier
    delta1 = delta1 + grad1 * multiplier
    delta2 = delta2 + grad2 * multiplier
    delta3 = delta3 + grad3 * multiplier

    # apply forces
    wp.atomic_sub(delta, i, delta0*w0*relaxation)
    wp.atomic_sub(delta, j, delta1*w1*relaxation)
    wp.atomic_sub(delta, k, delta2*w2*relaxation)
    wp.atomic_sub(delta, l, delta3*w3*relaxation)


@wp.kernel
def apply_deltas(x_orig: wp.array(dtype=wp.vec3),
                 v_orig: wp.array(dtype=wp.vec3),
                 x_pred: wp.array(dtype=wp.vec3),
                 delta: wp.array(dtype=wp.vec3),
                 dt: float,
                 x_out: wp.array(dtype=wp.vec3),
                 v_out: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    x0 = x_orig[tid]
    xp = x_pred[tid]

    # constraint deltas
    d = delta[tid]

    x_new = xp + d
    v_new = (x_new - x0)/dt

    x_out[tid] = x_new
    v_out[tid] = v_new

    # clear constraint deltas
    delta[tid] = wp.vec3(0.0)





@wp.func
def positional_correction(
    dx: wp.vec3,
    r1: wp.vec3,
    r2: wp.vec3,
    tf1: wp.transform,
    tf2: wp.transform,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    alpha_tilde: float,
    relaxation: float,
    lambda_in: float,
    max_lambda: float,
    deltas: wp.array(dtype=wp.spatial_vector),
    body_1: int,
    body_2: int,
) -> wp.vec3:
    # Computes and applies the correction impulse for a positional constraint.

    c = wp.length(dx)
    if c == 0.0:
        # print("c == 0.0 in positional correction")
        return wp.vec3(0.0, 0.0, 0.0)

    n = wp.normalize(dx)

    q1 = wp.transform_get_rotation(tf1)
    q2 = wp.transform_get_rotation(tf2)

    # Eq. 2-3 (make sure to project into the frame of the body)
    r1xn = wp.quat_rotate_inv(q1, wp.cross(r1, n))
    r2xn = wp.quat_rotate_inv(q2, wp.cross(r2, n))

    w1 = m_inv1 + wp.dot(r1xn, I_inv1 * r1xn)
    w2 = m_inv2 + wp.dot(r2xn, I_inv2 * r2xn)
    w = w1 + w2
    if w == 0.0:
        return wp.vec3(0.0, 0.0, 0.0)

    # Eq. 4-5
    d_lambda = (-c - alpha_tilde * lambda_in) / (w + alpha_tilde)
    lambda_out = lambda_in + d_lambda
    # print("d_lambda")
    # print(d_lambda)
    if max_lambda > 0.0 and wp.abs(lambda_out) > wp.abs(max_lambda):
    #     # print("lambda_out > max_lambda")
    #     # d_lambda = max_lambda
    #     # d_lambda = wp.min(d_lambda, max_lambda)
        return wp.vec3(0.0, 0.0, 0.0)
    p = d_lambda * n * relaxation

    # if wp.abs(d_lambda) > 0.1:
    #     return wp.vec3(0.0)

    if body_1 >= 0 and m_inv1 > 0.0:
        # Eq. 6
        dp = p

        # Eq. 8
        rd = wp.quat_rotate_inv(q1, wp.cross(r1, p))
        dq = wp.quat_rotate(q1, I_inv1 * rd) * 0.5
        w = wp.length(dq)
        if w > 0.01:
            dq = wp.normalize(dq) * 0.01
        wp.atomic_sub(deltas, body_1, wp.spatial_vector(dq, dp))

    if body_2 >= 0 and m_inv2 > 0.0:
        # Eq. 7
        dp = p

        # Eq. 9
        rd = wp.quat_rotate_inv(q2, wp.cross(r2, p))
        dq = wp.quat_rotate(q2, I_inv2 * rd) * 0.5
        w = wp.length(dq)
        if w > 0.01:
            dq = wp.normalize(dq) * 0.01
        wp.atomic_add(deltas, body_2, wp.spatial_vector(dq, dp))
    return p

@wp.func
def angular_correction(
    corr: wp.vec3,
    tf1: wp.transform,
    tf2: wp.transform,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    alpha_tilde: float,
    # lambda_prev: float,
    relaxation: float,
    deltas: wp.array(dtype=wp.spatial_vector),
    body_1: int,
    body_2: int,
) -> wp.vec3:
    # compute and apply the correction impulse for an angular constraint
    theta = wp.length(corr)
    if theta == 0.0:
        # print("theta == 0.0 in angular correction")
        return wp.vec3(0.0, 0.0, 0.0)
    n = wp.normalize(corr)

    q1 = wp.transform_get_rotation(tf1)
    q2 = wp.transform_get_rotation(tf2)

    # project variables to body rest frame as they are in local matrix
    n1 = wp.quat_rotate_inv(q1, n)
    n2 = wp.quat_rotate_inv(q2, n)

    # Eq. 11-12
    w1 = wp.dot(n1, I_inv1 * n1)
    w2 = wp.dot(n2, I_inv2 * n2)
    w = w1 + w2
    if w == 0.0:
        # print("w == 0.0 in angular correction")
        # # print("corr:")
        # # print(corr)
        # print("n1:")
        # print(n1)
        # print("n2:")
        # print(n2)
        # print("I_inv1:")
        # print(I_inv1)
        # print("I_inv2:")
        # print(I_inv2)
        return wp.vec3(0.0, 0.0, 0.0)

    # Eq. 13-14
    lambda_prev = 0.0
    d_lambda = (-theta - alpha_tilde * lambda_prev) / (w + alpha_tilde)
    # TODO consider lambda_prev?
    p = d_lambda * n * relaxation

    # Eq. 15-16
    dp = wp.vec3(0.0)
    if body_1 >= 0 and m_inv1 > 0.0:
        rd = n1 * d_lambda
        dq = wp.quat_rotate(q1, I_inv1 * rd) * relaxation
        w = wp.length(dq)
        if w > 0.01:
            dq = wp.normalize(dq) * 0.01
        wp.atomic_sub(deltas, body_1, wp.spatial_vector(dq, dp))
    if body_2 >= 0 and m_inv2 > 0.0:
        rd = n2 * d_lambda
        dq = wp.quat_rotate(q2, I_inv2 * rd) * relaxation
        w = wp.length(dq)
        if w > 0.01:
            dq = wp.normalize(dq) * 0.01
        wp.atomic_add(deltas, body_2, wp.spatial_vector(dq, dp))
    return p

@wp.kernel
def apply_body_deltas(
    q_in: wp.array(dtype=wp.transform),
    qd_in: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_I: wp.array(dtype=wp.mat33),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),
    deltas: wp.array(dtype=wp.spatial_vector),
    dt: float,
    q_out: wp.array(dtype=wp.transform),
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    inv_m = body_inv_m[tid]    
    if inv_m == 0.0:
        return
    inv_I = body_inv_I[tid]

    tf = q_in[tid]
    delta = deltas[tid]

    p0 = wp.transform_get_translation(tf)
    q0 = wp.transform_get_rotation(tf)

    x_com = p0 + wp.quat_rotate(q0, body_com[tid])

    dp = wp.spatial_bottom(delta) * inv_m
    dq = wp.spatial_top(delta)
    dq = wp.quat_rotate(q0, inv_I * wp.quat_rotate_inv(q0, dq))
    # dq = inv_I * wp.quat_rotate_inv(q, dq)
    # dq = wp.quat_rotate(q, inv_I * dq)

    # update position
    p = x_com + dp * dt * dt
    # p = p0 + dp * dt * dt

    # update orientation
    q = q0 + 0.5 * wp.quat(dq * dt * dt, 0.0) * q0

    # if wp.length(q) > 1.01:
    #     print("quaternion magnitude > 1 in apply_body_delta_positions")

    # if wp.length(q) < 0.98:
    #     print("quaternion magnitude < 1 in apply_body_delta_positions")

    q = wp.normalize(q)

    q_out[tid] = wp.transform(p - wp.quat_rotate(q, body_com[tid]), q)
    # q_out[tid] = wp.transform(p, q)

    v0 = wp.spatial_bottom(qd_in[tid])
    w0 = wp.spatial_top(qd_in[tid])
    # qd_out[tid] = wp.spatial_vector(w + dq * dt, v + dp * dt)



    
    # x_com = x0 + wp.quat_rotate(r0, body_com[tid])

    # # linear part
    v1 = v0 + dp * dt
    # x1 = x_com + v1 * dt

    # # x1 = x0 + v1 * dt

    # # angular part
    # # wb = wp.quat_rotate_inv(r0, w0)
    # # gyr = -(inv_inertia * wp.cross(wb, inertia*wb))
    # # w1 = w0 + dt * wp.quat_rotate(r0, gyr)

    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(q0, w0 + dq * dt)
    tb = -wp.cross(wb, body_I[tid]*wb)   # coriolis forces

    w1 = wp.quat_rotate(q0, wb + inv_I * tb * dt)
    # r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # # angular damping, todo: expose
    # # w1 = w1*(1.0-0.1*dt)

    # body_q_new[tid] = wp.transform(x1 - wp.quat_rotate(r1, body_com[tid]), r1)
    # # body_q_new[tid] = wp.transform(x1, r1)
    qd_out[tid] = wp.spatial_vector(w1, v1)

    # reset delta
    deltas[tid] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def apply_body_delta_velocities(
    qd_in: wp.array(dtype=wp.spatial_vector),
    deltas: wp.array(dtype=wp.spatial_vector),
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    qd_out[tid] = qd_in[tid] + deltas[tid]


# @wp.func
# def quat_basis_vector_a(q: wp.quat) -> wp.vec3:
#     x2 = q[0] * 2.0
#     w2 = q[3] * 2.0
#     return wp.vec3((q[3] * w2) - 1.0 + q[0] * x2, (q[2] * w2) + q[1] * x2, (-q[1] * w2) + q[2] * x2)

# @wp.func
# def quat_basis_vector_b(q: wp.quat) -> wp.vec3:
#     y2 = q[1] * 2.0
#     w2 = q[3] * 2.0
#     return wp.vec3((-q[2] * w2) + q[0] * y2, (q[3] * w2) - 1.0 + q[1] * y2, (q[0] * w2) + q[2] * y2)

# @wp.func
# def quat_basis_vector_c(q: wp.quat) -> wp.vec3:
#     z2 = q[2] * 2.0
#     w2 = q[3] * 2.0
#     return wp.vec3((q[1] * w2) + q[0] * z2, (-q[0] * w2) + q[1] * z2, (q[3] * w2) - 1.0 + q[2] * z2)



# decompose a quaternion into a sequence of 3 rotations around x,y',z' respectively, i.e.: q = q_z''q_y'q_x
@wp.func
def quat_decompose(q: wp.quat):

    R = wp.mat33(
            wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),
            wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),
            wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))

    # https://www.sedris.org/wg8home/Documents/WG80485.pdf
    phi = wp.atan2(R[1, 2], R[2, 2])
    theta = wp.asin(-R[0, 2])
    psi = wp.atan2(R[0, 1], R[0, 0])

    return -wp.vec3(phi, theta, psi)

@wp.kernel
def apply_joint_torques(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_act: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector)
):
    tid = wp.tid()
    type = joint_type[tid]
    if (type == wp.sim.JOINT_FIXED):
        return
    if (type == wp.sim.JOINT_FREE):
        return
    
    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]
    
    X_wp = X_pj
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    # parent transform and moment arm
    if (id_p >= 0):
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)
    
    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c
    com_c = body_com[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)    

    # local joint rotations
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # joint properties (for 1D joints)
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    axis = joint_axis[tid]
    act = joint_act[qd_start]

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    # handle angular constraints
    if (type == wp.sim.JOINT_REVOLUTE):
        a_p = wp.transform_vector(X_wp, axis)
        t_total += act * a_p
    elif (type == wp.sim.JOINT_PRISMATIC):
        a_p = wp.transform_vector(X_wp, axis)
        f_total += act * a_p
    elif (type == wp.sim.JOINT_COMPOUND):
        q_off = wp.transform_get_rotation(X_cj)
        q_pc = wp.quat_inverse(q_off)*wp.quat_inverse(q_p)*q_c*q_off
        # decompose to a compound rotation each axis 
        angles = quat_decompose(q_pc)

        # reconstruct rotation axes
        axis_0 = wp.vec3(1.0, 0.0, 0.0)
        q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        axis_2 = wp.quat_rotate(q_1*q_0, wp.vec3(0.0, 0.0, 1.0))

        q_w = q_p*q_off

        # joint dynamics
        t_total += joint_act[qd_start+0] * wp.quat_rotate(q_w, axis_0)
        t_total += joint_act[qd_start+1] * wp.quat_rotate(q_w, axis_1)
        t_total += joint_act[qd_start+2] * wp.quat_rotate(q_w, axis_2)
    elif (type == wp.sim.JOINT_UNIVERSAL):
        q_off = wp.transform_get_rotation(X_cj)
        q_pc = wp.quat_inverse(q_off)*wp.quat_inverse(q_p)*q_c*q_off
       
        # decompose to a compound rotation each axis 
        angles = quat_decompose(q_pc)

        # reconstruct rotation axes
        axis_0 = wp.vec3(1.0, 0.0, 0.0)
        q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        axis_2 = wp.quat_rotate(q_1*q_0, wp.vec3(0.0, 0.0, 1.0))

        q_w = q_p*q_off

        # free axes
        t_total += joint_act[qd_start+0] * wp.quat_rotate(q_w, axis_0)
        t_total += joint_act[qd_start+1] * wp.quat_rotate(q_w, axis_1)

    else:
        print("joint type not handled in apply_joint_torques")        
        
    # write forces
    if (id_p >= 0):
        wp.atomic_add(body_f, id_p, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total)) 
    wp.atomic_sub(body_f, id_c, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))


@wp.func
def quat_dof_limit(limit: float) -> float:
    # we cannot handle joint limits outside of [-2pi, 2pi]
    if wp.abs(limit) > 6.28318530718:
        return limit
    else:
        return wp.sin(0.5 * limit)

@wp.kernel
def solve_body_joints(body_q: wp.array(dtype=wp.transform),
                      body_qd: wp.array(dtype=wp.spatial_vector),
                      body_com: wp.array(dtype=wp.vec3),
                      body_inv_m: wp.array(dtype=float),
                      body_inv_I: wp.array(dtype=wp.mat33),
                      joint_q_start: wp.array(dtype=int),
                      joint_qd_start: wp.array(dtype=int),
                      joint_type: wp.array(dtype=int),
                      joint_parent: wp.array(dtype=int),
                      joint_child: wp.array(dtype=int),
                      joint_X_p: wp.array(dtype=wp.transform),
                      joint_X_c: wp.array(dtype=wp.transform),
                      joint_axis: wp.array(dtype=wp.vec3),
                      joint_target: wp.array(dtype=float),
                      joint_target_ke: wp.array(dtype=float),
                      joint_target_kd: wp.array(dtype=float),
                      joint_limit_lower: wp.array(dtype=float),
                      joint_limit_upper: wp.array(dtype=float),
                      joint_twist_lower: wp.array(dtype=float),
                      joint_twist_upper: wp.array(dtype=float),
                      joint_linear_compliance: wp.array(dtype=float),
                      joint_angular_compliance: wp.array(dtype=float),
                      angular_relaxation: float,
                      positional_relaxation: float,
                      dt: float,
                      deltas: wp.array(dtype=wp.spatial_vector)):
    tid = wp.tid()
    type = joint_type[tid]

    if (type == wp.sim.JOINT_FREE):
        return
    
    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]
    
    X_wp = X_pj
    m_inv_p = 0.0
    I_inv_p = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    vel_p = wp.vec3(0.0)
    omega_p = wp.vec3(0.0)
    # parent transform and moment arm
    if (id_p >= 0):
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
        m_inv_p = body_inv_m[id_p]
        I_inv_p = body_inv_I[id_p]
        vel_p = wp.spatial_bottom(body_qd[id_p])
        omega_p = wp.spatial_top(body_qd[id_p])
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)
    
    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c  # note we do not apply X_cj here (it is used in multi-dof joints)
    com_c = body_com[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)    
    m_inv_c = body_inv_m[id_c]
    I_inv_c = body_inv_I[id_c]
    vel_c = wp.spatial_bottom(body_qd[id_c])
    omega_c = wp.spatial_top(body_qd[id_c])

    if m_inv_p == 0.0 and m_inv_c == 0.0:
        # connection between two immovable bodies
        return

    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    rel_pose = wp.transform_inverse(X_wp) * X_wc
    rel_p = wp.transform_get_translation(rel_pose)
    frame_p = wp.quat_to_matrix(wp.transform_get_rotation(X_wp))
    
    axis = joint_axis[tid]
    linear_compliance = joint_linear_compliance[tid]
    angular_compliance = joint_angular_compliance[tid]

    lower_pos_limits = wp.vec3(0.0)
    upper_pos_limits = wp.vec3(0.0)
    target_pos_ke = wp.vec3(0.0)
    target_pos_kd = wp.vec3(0.0)
    target_pos = wp.vec3(0.0)
    if (type == wp.sim.JOINT_PRISMATIC):
        lo = axis * joint_limit_lower[qd_start]
        up = axis * joint_limit_upper[qd_start]
        lower_pos_limits = wp.vec3(
            wp.min(lo[0], up[0]),
            wp.min(lo[1], up[1]),
            wp.min(lo[2], up[2]))
        upper_pos_limits = wp.vec3(
            wp.max(lo[0], up[0]),
            wp.max(lo[1], up[1]),
            wp.max(lo[2], up[2]))        
        target_pos_ke = axis * joint_target_ke[qd_start]
        target_pos_kd = axis * joint_target_kd[qd_start]
        target_pos = axis * joint_target[qd_start]

    # joint connection points
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    # handle positional constraints
    for dim in range(3):
        err = rel_p[dim]

        lower = lower_pos_limits[dim]
        upper = upper_pos_limits[dim]

        compliance = linear_compliance
        damping = 0.0
        if wp.abs(target_pos_ke[dim]) > 0.0:
            err -= target_pos[dim]
            compliance = 1.0 / wp.abs(target_pos_ke[dim])
            damping = wp.abs(target_pos_kd[dim])
        if err < lower:
            err = rel_p[dim] - lower
            compliance = linear_compliance
            damping = 0.0
        elif err > upper:
            err = rel_p[dim] - upper
            compliance = linear_compliance
            damping = 0.0
        else:
            err = 0.0

        if wp.abs(err) > 1e-9:
            # compute gradients
            linear_c = wp.vec3(frame_p[0, dim], frame_p[1, dim], frame_p[2, dim])
            linear_p = -linear_c
            # note that x_c appearing in both is correct
            r_p = x_c - wp.transform_point(pose_p, com_p)
            r_c = x_c - wp.transform_point(pose_c, com_c)
            angular_p = -wp.cross(r_p, linear_c)
            angular_c = wp.cross(r_c, linear_c)
            # constraint time derivative
            derr = wp.dot(linear_p, vel_p) + wp.dot(linear_c, vel_c) + wp.dot(angular_p, omega_p) + wp.dot(angular_c, omega_c)
            
            lambda_in = 0.0
            d_lambda = compute_positional_correction(
                err, derr, pose_p, pose_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
                linear_p, linear_c, angular_p, angular_c, lambda_in, compliance, damping, dt)
            # d_lambda = compute_positional_correction(
            #     err, derr, X_wp, X_wc, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
            #     linear_p, linear_c, angular_p, angular_c, lambda_in, compliance, damping, dt)

            lin_delta_p += linear_p * (d_lambda * positional_relaxation)
            ang_delta_p += angular_p * (d_lambda * positional_relaxation)
            lin_delta_c += linear_c * (d_lambda * angular_relaxation)
            ang_delta_c += angular_c * (d_lambda * angular_relaxation)


    # local joint rotations
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # make quats lie in same hemisphere
    if (wp.dot(q_p, q_c) < 0.0):
        q_c *= -1.0

    # handle angular constraints
    rel_q = wp.quat_inverse(q_p) * q_c
    
    qtwist = wp.normalize(wp.quat(rel_q[0], 0.0, 0.0, rel_q[3]))
    qswing = rel_q*wp.quat_inverse(qtwist)
    errs = wp.vec3(qtwist[0], qswing[1], qswing[2])
        
    s = wp.sqrt(rel_q[0]*rel_q[0] + rel_q[3]*rel_q[3])			
    invs = 1.0/s
    invscube = invs*invs*invs

    lower_ang_limits = wp.vec3(0.0)
    upper_ang_limits = wp.vec3(0.0)
    target_ang_ke = wp.vec3(0.0)
    target_ang_kd = wp.vec3(0.0)
    target_ang = wp.vec3(0.0)
    
    if (type == wp.sim.JOINT_REVOLUTE):
        # convert position limits/targets to quaternion space
        lo = axis * quat_dof_limit(joint_limit_lower[qd_start])
        up = axis * quat_dof_limit(joint_limit_upper[qd_start])
        lower_ang_limits = wp.vec3(
            wp.min(lo[0], up[0]),
            wp.min(lo[1], up[1]),
            wp.min(lo[2], up[2]))
        upper_ang_limits = wp.vec3(
            wp.max(lo[0], up[0]),
            wp.max(lo[1], up[1]),
            wp.max(lo[2], up[2]))
        
        target_ang_ke = axis * joint_target_ke[qd_start]
        target_ang_kd = axis * joint_target_kd[qd_start]
        target_ang = axis * quat_dof_limit(joint_target[qd_start])
    elif (type == wp.sim.JOINT_UNIVERSAL):
        q_off = wp.transform_get_rotation(X_cj)
        mat = wp.quat_to_matrix(q_off)
        axis_0 = wp.vec3(mat[0, 0], mat[1, 0], mat[2, 0])
        axis_1 = wp.vec3(mat[0, 1], mat[1, 1], mat[2, 1])
        
        lower_0 = quat_dof_limit(joint_limit_lower[qd_start])
        upper_0 = quat_dof_limit(joint_limit_upper[qd_start])
        lower_1 = quat_dof_limit(joint_limit_lower[qd_start+1])
        upper_1 = quat_dof_limit(joint_limit_upper[qd_start+1])

        # find dof limits while considering negative axis dimensions and joint limits
        lo0 = axis_0 * lower_0
        up0 = axis_0 * upper_0
        lo1 = axis_1 * lower_1
        up1 = axis_1 * upper_1
        lower_ang_limits = wp.vec3(
            wp.min(wp.min(lo0[0], up0[0]), wp.min(lo1[0], up1[0])),
            wp.min(wp.min(lo0[1], up0[1]), wp.min(lo1[1], up1[1])), 
            wp.min(wp.min(lo0[2], up0[2]), wp.min(lo1[2], up1[2])))
        upper_ang_limits = wp.vec3(
            wp.max(wp.max(lo0[0], up0[0]), wp.max(lo1[0], up1[0])),
            wp.max(wp.max(lo0[1], up0[1]), wp.max(lo1[1], up1[1])), 
            wp.max(wp.max(lo0[2], up0[2]), wp.max(lo1[2], up1[2])))
        
        ke_0 = joint_target_ke[qd_start]
        kd_0 = joint_target_kd[qd_start]
        ke_1 = joint_target_ke[qd_start+1]
        kd_1 = joint_target_kd[qd_start+1]
        ke_sum = ke_0 + ke_1
        # count how many dofs have non-zero stiffness
        ke_dofs = wp.nonzero(ke_0) + wp.nonzero(ke_1)
        if ke_sum > 0.0:
            # XXX we take the average stiffness, damping per dof
            target_ang_ke = axis_0 * (ke_0/ke_dofs) + axis_1 * (ke_1/ke_dofs)
            target_ang_kd = axis_0 * (kd_0/ke_dofs) + axis_1 * (kd_1/ke_dofs)
            ang_0 = quat_dof_limit(joint_target[qd_start]) * ke_0 / ke_sum
            ang_1 = quat_dof_limit(joint_target[qd_start+1]) * ke_1 / ke_sum
            target_ang = axis_0 * ang_0 + axis_1 * ang_1
    elif (type == wp.sim.JOINT_COMPOUND):
        q_off = wp.transform_get_rotation(X_cj)
        mat = wp.quat_to_matrix(q_off)
        axis_0 = wp.vec3(mat[0, 0], mat[1, 0], mat[2, 0])
        axis_1 = wp.vec3(mat[0, 1], mat[1, 1], mat[2, 1])
        axis_2 = wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
        
        lower_0 = quat_dof_limit(joint_limit_lower[qd_start])
        upper_0 = quat_dof_limit(joint_limit_upper[qd_start])
        lower_1 = quat_dof_limit(joint_limit_lower[qd_start+1])
        upper_1 = quat_dof_limit(joint_limit_upper[qd_start+1])
        lower_2 = quat_dof_limit(joint_limit_lower[qd_start+2])
        upper_2 = quat_dof_limit(joint_limit_upper[qd_start+2])

        # find dof limits while considering negative axis dimensions and joint limits
        lo0 = axis_0 * lower_0
        up0 = axis_0 * upper_0
        lo1 = axis_1 * lower_1
        up1 = axis_1 * upper_1
        lo2 = axis_2 * lower_2
        up2 = axis_2 * upper_2
        lower_ang_limits = wp.vec3(
            wp.min(wp.min(wp.min(lo0[0], up0[0]), wp.min(lo1[0], up1[0])), wp.min(lo2[0], up2[0])),
            wp.min(wp.min(wp.min(lo0[1], up0[1]), wp.min(lo1[1], up1[1])), wp.min(lo2[1], up2[1])), 
            wp.min(wp.min(wp.min(lo0[2], up0[2]), wp.min(lo1[2], up1[2])), wp.min(lo2[2], up2[2])))
        upper_ang_limits = wp.vec3(
            wp.max(wp.max(wp.max(lo0[0], up0[0]), wp.max(lo1[0], up1[0])), wp.max(lo2[0], up2[0])),
            wp.max(wp.max(wp.max(lo0[1], up0[1]), wp.max(lo1[1], up1[1])), wp.max(lo2[1], up2[1])), 
            wp.max(wp.max(wp.max(lo0[2], up0[2]), wp.max(lo1[2], up1[2])), wp.max(lo2[2], up2[2])))
        
        ke_0 = joint_target_ke[qd_start]
        kd_0 = joint_target_kd[qd_start]
        ke_1 = joint_target_ke[qd_start+1]
        kd_1 = joint_target_kd[qd_start+1]
        ke_2 = joint_target_ke[qd_start+2]
        kd_2 = joint_target_kd[qd_start+2]
        ke_sum = ke_0 + ke_1 + ke_2
        # count how many dofs have non-zero stiffness
        ke_dofs = wp.nonzero(ke_0) + wp.nonzero(ke_1) + wp.nonzero(ke_2)
        if ke_sum > 0.0:
            # XXX we take the average stiffness, damping per dof
            target_ang_ke = axis_0 * (ke_0/ke_dofs) + axis_1 * (ke_1/ke_dofs) + axis_2 * (ke_2/ke_dofs)
            target_ang_kd = axis_0 * (kd_0/ke_dofs) + axis_1 * (kd_1/ke_dofs) + axis_2 * (kd_2/ke_dofs)
            ang_0 = quat_dof_limit(joint_target[qd_start]) * ke_0 / ke_sum
            ang_1 = quat_dof_limit(joint_target[qd_start+1]) * ke_1 / ke_sum
            ang_2 = quat_dof_limit(joint_target[qd_start+2]) * ke_2 / ke_sum
            target_ang = axis_0 * ang_0 + axis_1 * ang_1 + axis_2 * ang_2
    

    if (type == wp.sim.JOINT_BALL):
        if (joint_limit_lower[qd_start] != 0.0 or joint_limit_upper[qd_start] != 0.0 or joint_target_ke[qd_start] != 0.0):
            print("Warning: ball joints with position limits or target stiffness are not yet supported!")
    else:
        for dim in range(3):
            err = 0.0
         
            lower = lower_ang_limits[dim]
            upper = upper_ang_limits[dim]

            compliance = angular_compliance
            damping = 0.0
            if wp.abs(target_ang_ke[dim]) > 0.0:
                err = errs[dim] - target_ang[dim]
                compliance = 1.0 / wp.abs(target_ang_ke[dim])
                damping = wp.abs(target_ang_kd[dim])
            if errs[dim] < lower:
                err = errs[dim] - lower
                compliance = angular_compliance
                damping = 0.0
            elif errs[dim] > upper:
                err = errs[dim] - upper
                compliance = angular_compliance
                damping = 0.0

            if wp.abs(err) > 1e-9:
                # analytic gradients of swing-twist decomposition
                if dim == 0:
                    grad = wp.quat(1.0*invs - rel_q[0]*rel_q[0]*invscube, 0.0, 0.0, -(rel_q[3]*rel_q[0])*invscube)
                elif dim == 1:
                    grad = wp.quat(-rel_q[3]*(rel_q[3]*rel_q[2] + rel_q[0]*rel_q[1])*invscube, rel_q[3]*invs, -rel_q[0]*invs, rel_q[0]*(rel_q[3]*rel_q[2] + rel_q[0]*rel_q[1])*invscube)
                else:
                    grad = wp.quat(rel_q[3]*(rel_q[3]*rel_q[1] - rel_q[0]*rel_q[2])*invscube, rel_q[0]*invs, rel_q[3]*invs, rel_q[0]*(rel_q[2]*rel_q[0] - rel_q[3]*rel_q[1])*invscube)
                
                quat_c = 0.5*q_p*grad* wp.quat_inverse(q_c)
                angular_c = wp.vec3(quat_c[0], quat_c[1], quat_c[2])
                angular_p = -angular_c
                # time derivative of the constraint
                derr = wp.dot(angular_p, omega_p) + wp.dot(angular_c, omega_c)

                d_lambda = compute_angular_correction(
                    err, derr, pose_p, pose_c, I_inv_p, I_inv_c,
                    angular_p, angular_c, 0.0, compliance, damping, dt) * angular_relaxation
                # d_lambda = compute_angular_correction(
                #     err, derr, X_wp, X_wc, I_inv_p, I_inv_c,
                #     angular_p, angular_c, 0.0, compliance, damping, dt) * angular_relaxation
                # update deltas
                ang_delta_p += angular_p * d_lambda
                ang_delta_c += angular_c * d_lambda

    if (id_p >= 0):
        wp.atomic_add(deltas, id_p, wp.spatial_vector(ang_delta_p, lin_delta_p))
    if (id_c >= 0):
        wp.atomic_add(deltas, id_c, wp.spatial_vector(ang_delta_c, lin_delta_c))

@wp.func
def compute_contact_constraint_delta(
    err: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    m_inv_a: float,
    m_inv_b: float,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    linear_a: wp.vec3, 
    linear_b: wp.vec3, 
    angular_a: wp.vec3, 
    angular_b: wp.vec3, 
    inv_weight_a: float,
    inv_weight_b: float,
    relaxation: float,
    dt: float
) -> float:
    denom = 0.0
    denom += wp.length_sq(linear_a)*m_inv_a*inv_weight_a
    denom += wp.length_sq(linear_b)*m_inv_b*inv_weight_b

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)*inv_weight_a
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)*inv_weight_b

    deltaLambda = -err / (dt*dt*denom)

    return deltaLambda*relaxation


@wp.func
def compute_constraint_delta(
    err: float,
    derr: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    m_inv_a: float,
    m_inv_b: float,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    linear_a: wp.vec3, 
    linear_b: wp.vec3, 
    angular_a: wp.vec3, 
    angular_b: wp.vec3, 
    lambda_in: float,
    compliance: float,
    damping: float,
    relaxation: float,
    dt: float
) -> float:
    denom = 0.0
    denom += wp.length_sq(linear_a)*m_inv_a
    denom += wp.length_sq(linear_b)*m_inv_b

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    alpha = compliance
    gamma = compliance * damping

    deltaLambda = -(err + alpha*lambda_in + gamma*derr) / (dt*(dt + gamma)*denom + alpha)

    return deltaLambda*relaxation


@wp.func
def compute_angular_correction_3d(
    corr: wp.vec3,
    q1: wp.quat,
    q2: wp.quat,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    alpha_tilde: float,
    # lambda_prev: float,
    relaxation: float,
    dt: float,
):
    # compute and apply the correction impulse for an angular constraint
    theta = wp.length(corr)
    if theta == 0.0:
        return 0.0

    n = wp.normalize(corr)

    # project variables to body rest frame as they are in local matrix
    n1 = wp.quat_rotate_inv(q1, n)
    n2 = wp.quat_rotate_inv(q2, n)

    # Eq. 11-12
    w1 = wp.dot(n1, I_inv1 * n1)
    w2 = wp.dot(n2, I_inv2 * n2)
    w = w1 + w2
    if w == 0.0:
        return 0.0
    #     # print("w == 0.0 in angular correction")
    #     # # print("corr:")
    #     # # print(corr)
    #     # print("n1:")
    #     # print(n1)
    #     # print("n2:")
    #     # print(n2)
    #     # print("I_inv1:")
    #     # print(I_inv1)
    #     # print("I_inv2:")
    #     # print(I_inv2)
    #     return wp.vec3(0.0, 0.0, 0.0)

    # Eq. 13-14
    lambda_prev = 0.0
    d_lambda = (-theta - alpha_tilde * lambda_prev) / (w * dt * dt + alpha_tilde)
    # TODO consider lambda_prev?
    # p = d_lambda * n * relaxation

    # Eq. 15-16
    return d_lambda * relaxation

@wp.func
def compute_positional_correction(
    err: float,
    derr: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    m_inv_a: float,
    m_inv_b: float,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    linear_a: wp.vec3, 
    linear_b: wp.vec3, 
    angular_a: wp.vec3, 
    angular_b: wp.vec3, 
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float
) -> float:
    denom = 0.0
    denom += wp.length_sq(linear_a)*m_inv_a
    denom += wp.length_sq(linear_b)*m_inv_b

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    alpha = compliance
    gamma = compliance * damping

    deltaLambda = -(err + alpha*lambda_in + gamma*derr) / (dt*(dt + gamma)*denom + alpha)

    return deltaLambda



@wp.func
def compute_angular_correction(
    err: float,
    derr: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    angular_a: wp.vec3, 
    angular_b: wp.vec3, 
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float
) -> float:
    denom = 0.0

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    alpha = compliance
    gamma = compliance * damping

    deltaLambda = -(err + alpha*lambda_in + gamma*derr) / (dt*(dt + gamma)*denom + alpha)

    return deltaLambda


@wp.kernel
def update_body_contact_weights(
    body_q: wp.array(dtype=wp.transform),
    contact_count: wp.array(dtype=int),
    contact_body0: wp.array(dtype=int),
    contact_body1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_thickness: wp.array(dtype=float),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    shape_X_co: wp.array(dtype=wp.transform),
    # outputs
    contact_inv_weight: wp.array(dtype=float),
    active_contact_point0: wp.array(dtype=wp.vec3),
    active_contact_point1: wp.array(dtype=wp.vec3),
    active_contact_distance: wp.array(dtype=float),
):

    tid = wp.tid()

    count = contact_count[0]
    if (tid >= count):
        return
        
    body_a = contact_body0[tid]
    body_b = contact_body1[tid]

    # body position in world space
    thickness = contact_thickness[tid]
    n = contact_normal[tid]
    bx_a = contact_point0[tid]
    bx_b = contact_point1[tid]
    # body to world transform
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()

    if (body_a >= 0):
        X_wb_a = body_q[body_a]
    if (body_b >= 0):
        X_wb_b = body_q[body_b]
    
    bx_a = wp.transform_point(X_wb_a, bx_a)
    bx_b = wp.transform_point(X_wb_b, bx_b)
    
    n = contact_normal[tid]
    d = -wp.dot(n, bx_b-bx_a) - thickness
    active_contact_distance[tid] = d

    if d < 0.0:
        if (body_a >= 0):
            wp.atomic_add(contact_inv_weight, body_a, 1.0)
        if (body_b >= 0):
            wp.atomic_add(contact_inv_weight, body_b, 1.0)
        active_contact_point0[tid] = bx_a
        active_contact_point1[tid] = bx_b


@wp.kernel
def solve_body_contact_positions(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    contact_count: wp.array(dtype=int),
    contact_body0: wp.array(dtype=int),
    contact_body1: wp.array(dtype=int),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    active_contact_point0: wp.array(dtype=wp.vec3),
    active_contact_point1: wp.array(dtype=wp.vec3),
    active_contact_distance: wp.array(dtype=float),
    shape_materials: ShapeContactMaterial,
    contact_inv_weight: wp.array(dtype=float),
    relaxation: float,
    dt: float,
    max_penetration: float,
    contact_torsional_friction: float,
    contact_rolling_friction: float,
    # outputs
    deltas: wp.array(dtype=wp.spatial_vector),
):
    
    tid = wp.tid()

    count = contact_count[0]
    if (tid >= count):
        return
    d = active_contact_distance[tid]
    if d >= 0.0:
        return
        
    body_a = contact_body0[tid]
    body_b = contact_body1[tid]

    n = contact_normal[tid]
    
    m_inv_a = 0.0
    m_inv_b = 0.0
    I_inv_a = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_b = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # center of mass in body frame
    X_com_a = wp.vec3(0.0)
    X_com_b = wp.vec3(0.0)
    # moment arm in world frame
    r_a = wp.vec3(0.0)
    r_b = wp.vec3(0.0)
    # body to world transform
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    # angular velocities
    omega_a = wp.vec3(0.0)
    omega_b = wp.vec3(0.0)
    # contact offset in body frame
    offset_a = contact_offset0[tid]
    offset_b = contact_offset1[tid]
    # contact constraint weights
    inv_weight_a = 1.0
    inv_weight_b = 1.0

    if (body_a >= 0):
        X_wb_a = body_q[body_a]
        X_com_a = body_com[body_a]
        m_inv_a = body_m_inv[body_a]
        I_inv_a = body_I_inv[body_a]
        inv_weight_a = contact_inv_weight[body_a]
        omega_a = wp.spatial_top(body_qd[body_a])

    if (body_b >= 0):
        X_wb_b = body_q[body_b]
        X_com_b = body_com[body_b]
        m_inv_b = body_m_inv[body_b]
        I_inv_b = body_I_inv[body_b]
        inv_weight_b = contact_inv_weight[body_b]
        omega_b = wp.spatial_top(body_qd[body_b])
    
    # use average contact material properties
    mat_nonzero = 0
    mu = 0.0
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if (shape_a >= 0):
        mat_nonzero += 1
        mu += shape_materials.mu[shape_a]
    if (shape_b >= 0):
        mat_nonzero += 1
        mu += shape_materials.mu[shape_b]
    if (mat_nonzero > 0):
        mu /= float(mat_nonzero)

    
    bx_a = active_contact_point0[tid]
    bx_b = active_contact_point1[tid]

    r_a = bx_a - wp.transform_point(X_wb_a, X_com_a)
    r_b = bx_b - wp.transform_point(X_wb_b, X_com_b)

    angular_a = -wp.cross(r_a, n)
    angular_b = wp.cross(r_b, n)

    # limit penetration to prevent extreme constraint deltas
    d = wp.max(max_penetration, d)
    lambda_n = compute_contact_constraint_delta(
        d, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b,
        -n, n, angular_a, angular_b, inv_weight_a, inv_weight_b, relaxation, dt)

    lin_delta_a = -n * lambda_n
    lin_delta_b = n * lambda_n
    ang_delta_a = angular_a * lambda_n
    ang_delta_b = angular_b * lambda_n

    # linear friction
    if (mu > 0.0):

        # add on displacement from surface offsets, this ensures we include any rotational effects due to thickness from feature
        # need to use the current rotation to account for friction due to angular effects (e.g.: slipping contact)
        bx_a += wp.transform_vector(X_wb_a, offset_a)
        bx_b += wp.transform_vector(X_wb_b, offset_b)

        pos_a = bx_a
        pos_b = bx_b

        # update delta
        delta = pos_b-pos_a
        frictionDelta = -(delta - wp.dot(n, delta)*n)

        perp = wp.normalize(frictionDelta)

        r_a = pos_a - wp.transform_point(X_wb_a, X_com_a)
        r_b = pos_b - wp.transform_point(X_wb_b, X_com_b)
        
        angular_a = -wp.cross(r_a, perp)
        angular_b = wp.cross(r_b, perp)

        err = wp.length(frictionDelta)

        if (err > 0.0):
            lambda_fr = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b,
                -perp, perp, angular_a, angular_b, inv_weight_a, inv_weight_b, 1.0, dt)

            # limit friction based on incremental normal force, good approximation to limiting on total force
            lambda_fr = wp.max(lambda_fr, -lambda_n*mu)

            lin_delta_a -= perp*lambda_fr
            lin_delta_b += perp*lambda_fr

            ang_delta_a += angular_a*lambda_fr
            ang_delta_b += angular_b*lambda_fr

    torsional_friction = mu * contact_torsional_friction

    delta_omega = omega_b - omega_a

    if (torsional_friction > 0.0):
        err = wp.dot(delta_omega, n)*dt

        if (wp.abs(err) > 0.0):
            lin = wp.vec3(0.0)
            lambda_torsion = compute_contact_constraint_delta(err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b,
            lin, lin, -n, n, inv_weight_a, inv_weight_b, 1.0, dt)

            lambda_torsion = wp.clamp(lambda_torsion, -lambda_n*torsional_friction, lambda_n*torsional_friction)
            
            ang_delta_a += n*lambda_torsion
            ang_delta_b -= n*lambda_torsion
    
    rolling_friction = mu * contact_rolling_friction
    if (rolling_friction > 0.0):
        delta_omega -= wp.dot(n, delta_omega)*n
        err = wp.length(delta_omega)*dt
        if (err > 0.0):
            lin = wp.vec3(0.0)
            roll_n = wp.normalize(delta_omega)
            lambda_roll = compute_contact_constraint_delta(err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b,
            lin, lin, -roll_n, roll_n, inv_weight_a, inv_weight_b, 1.0, dt)

            lambda_roll = wp.max(lambda_roll, -lambda_n*rolling_friction)
            
            ang_delta_a += roll_n*lambda_roll
            ang_delta_b -= roll_n*lambda_roll

    if (body_a >= 0):
        wp.atomic_sub(deltas, body_a, wp.spatial_vector(ang_delta_a, lin_delta_a))
    if (body_b >= 0):
        wp.atomic_sub(deltas, body_b, wp.spatial_vector(ang_delta_b, lin_delta_b))


@wp.kernel
def update_body_velocities(
    poses: wp.array(dtype=wp.transform),
    poses_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    dt: float,
    qd_out: wp.array(dtype=wp.spatial_vector)
):
    tid = wp.tid()

    pose = poses[tid]
    pose_prev = poses_prev[tid]

    x = wp.transform_get_translation(pose)
    x_prev = wp.transform_get_translation(pose_prev)

    q = wp.transform_get_rotation(pose)
    q_prev = wp.transform_get_rotation(pose_prev)

    # Update body velocities according to Alg. 2
    # XXX we consider the body COM as the origin of the body frame
    x_com = x + wp.quat_rotate(q, body_com[tid])
    x_com_prev = x_prev + wp.quat_rotate(q_prev, body_com[tid])

    # XXX consider the velocity of the COM
    v = (x_com - x_com_prev) / dt
    dq = q * wp.quat_inverse(q_prev)

    omega = 2.0/dt * wp.vec3(dq[0], dq[1], dq[2])
    if dq[3] < 0.0:
        omega = -omega

    qd_out[tid] = wp.spatial_vector(omega, v)

@wp.kernel
def apply_rigid_restitution(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    contact_count: wp.array(dtype=int),
    contact_body0: wp.array(dtype=int),
    contact_body1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    shape_materials: ShapeContactMaterial,
    active_contact_distance: wp.array(dtype=float),
    contact_inv_weight: wp.array(dtype=float),
    gravity: wp.array(dtype=float),
    bounce_threshold: float,
    dt: float,
    # outputs
    deltas: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = contact_count[0]
    if (tid >= count):
        return
    d = active_contact_distance[tid]
    if d >= 0.0:
        return
    
    # use average contact material properties
    mat_nonzero = 0
    restitution = 0.0
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if (shape_a >= 0):
        mat_nonzero += 1
        restitution += shape_materials.restitution[shape_a]
    if (shape_b >= 0):
        mat_nonzero += 1
        restitution += shape_materials.restitution[shape_b]
    if (mat_nonzero > 0):
        restitution /= float(mat_nonzero)
    if (restitution <= 0.0):
        return
        
    body_a = contact_body0[tid]
    body_b = contact_body1[tid]
    
    m_inv_a = 0.0
    m_inv_b = 0.0
    I_inv_a = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_b = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # body to world transform
    X_wb_a_prev = wp.transform_identity()
    X_wb_b_prev = wp.transform_identity()
    # previous velocity at contact points
    v_a = wp.vec3(0.0)
    v_b = wp.vec3(0.0)
    # new velocity at contact points
    v_a_new = wp.vec3(0.0)
    v_b_new = wp.vec3(0.0)
    # inverse mass used to compute the impulse
    inv_mass = 0.0

    if (body_a >= 0):
        X_wb_a_prev = body_q_prev[body_a]
        X_wb_a = body_q[body_a]
        m_inv_a = body_m_inv[body_a]
        I_inv_a = body_I_inv[body_a]

    if (body_b >= 0):
        X_wb_b_prev = body_q_prev[body_b]
        X_wb_b = body_q[body_b]
        m_inv_b = body_m_inv[body_b]
        I_inv_b = body_I_inv[body_b]
    
    r_a_local = contact_point0[tid] + contact_offset0[tid]
    r_b_local = contact_point1[tid] + contact_offset1[tid]
    r_a = wp.transform_vector(X_wb_a_prev, r_a_local)
    r_b = wp.transform_vector(X_wb_b_prev, r_b_local)
    
    n = contact_normal[tid]
    g = wp.vec3(gravity[0], gravity[1], gravity[2])
    if (body_a >= 0):
        v_a = velocity_at_point(body_qd_prev[body_a], r_a) + g*dt
        v_a_new = velocity_at_point(body_qd[body_a], r_a)
        q_a = wp.transform_get_rotation(X_wb_a_prev)
        rxn = wp.quat_rotate_inv(q_a, wp.cross(r_a, n))
        # inv_mass += contact_inv_weight[body_a] * (m_inv_a + wp.dot(rxn, I_inv_a * rxn))
        inv_mass += (m_inv_a + wp.dot(rxn, I_inv_a * rxn))
    if (body_b >= 0):
        v_b = velocity_at_point(body_qd_prev[body_b], r_b) + g*dt
        v_b_new = velocity_at_point(body_qd[body_b], r_b)
        q_b = wp.transform_get_rotation(X_wb_b_prev)
        rxn = wp.quat_rotate_inv(q_b, wp.cross(r_b, n))
        # inv_mass += contact_inv_weight[body_b] * (m_inv_b + wp.dot(rxn, I_inv_b * rxn))
        inv_mass += (m_inv_b + wp.dot(rxn, I_inv_b * rxn))

    rel_vel_old = wp.dot(n, v_a - v_b)

    if (wp.abs(rel_vel_old) < bounce_threshold):
        # print("rel_vel_old < bounce_threshold")
        return

    rel_vel_new = wp.dot(n, v_a_new - v_b_new)
    j = -(rel_vel_new + restitution*rel_vel_old)
    # j = -(rel_vel_new + restitution*rel_vel_old) / inv_mass    
    p = n*j
    # print(rel_vel_new)
    if (body_a >= 0):
        q_a = wp.transform_get_rotation(X_wb_a)
        rxp = wp.quat_rotate_inv(q_a, wp.cross(r_a, p))
        dq = wp.quat_rotate(q_a, I_inv_a * rxp)
        # wp.atomic_add(deltas, body_a, wp.spatial_vector(dq, m_inv_a/contact_inv_weight[body_a]*p))
        wp.atomic_add(deltas, body_a, wp.spatial_vector(dq, p / contact_inv_weight[body_a]))

    if (body_b >= 0):
        q_b = wp.transform_get_rotation(X_wb_b)
        rxp = wp.quat_rotate_inv(q_b, wp.cross(r_b, p))
        dq = wp.quat_rotate(q_b, I_inv_b * rxp)
        # wp.atomic_sub(deltas, body_b, wp.spatial_vector(dq, m_inv_b/contact_inv_weight[body_b]*p))
        wp.atomic_sub(deltas, body_b, wp.spatial_vector(dq, p / contact_inv_weight[body_b]))


class XPBDIntegrator:
    """A implicit integrator using XPBD

    After constructing `Model` and `State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Semi-implicit time integration is a variational integrator that 
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example:

        >>> integrator = wp.SemiImplicitIntegrator()
        >>>
        >>> # simulation loop
        >>> for i in range(100):
        >>>     state = integrator.forward(model, state, dt)

    """

    def __init__(self,
                 iterations=2,
                 soft_body_relaxation=1.0,
                 joint_positional_relaxation=1.0,
                 joint_angular_relaxation=0.4,
                 contact_normal_relaxation=1.0,
                 contact_friction_relaxation=1.0,
                 contact_con_weighting=True,
                 max_rigid_contact_penetration=-0.1,
                 angular_damping=0.05):

        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation

        self.joint_positional_relaxation = joint_positional_relaxation
        self.joint_angular_relaxation = joint_angular_relaxation

        self.contact_normal_relaxation = contact_normal_relaxation
        self.contact_friction_relaxation = contact_friction_relaxation

        self.contact_con_weighting = contact_con_weighting

        # maximum penetration depth to be used for contact handling in order
        # to clip the contact impulse to reasonable levels in case of strong
        # interpenetrations between rigid bodies (has to be a value < 0)
        self.max_rigid_contact_penetration = max_rigid_contact_penetration

        self.angular_damping = angular_damping

    def simulate(self, model, state_in, state_out, dt):

        with wp.ScopedTimer("simulate", False):

            particle_q = None
            particle_qd = None

            if (model.particle_count):
                particle_q = wp.zeros_like(state_in.particle_q)
                particle_qd = wp.zeros_like(state_in.particle_qd)

                # alloc particle force buffer
                state_out.particle_f.zero_()

            if (not self.contact_con_weighting):
                model.rigid_contact_inv_weight.zero_()

            # ----------------------------
            # integrate particles

            if (model.particle_count):
                wp.launch(kernel=integrate_particles,
                          dim=model.particle_count,
                          inputs=[
                              state_in.particle_q,
                              state_in.particle_qd,
                              state_out.particle_f,
                              model.particle_inv_mass,
                              model.gravity,
                              dt
                          ],
                          outputs=[
                              particle_q,
                              particle_qd],
                          device=model.device)

            for i in range(self.iterations):

                # damped springs
                if (model.spring_count):

                    wp.launch(kernel=solve_springs,
                              dim=model.spring_count,
                              inputs=[
                                  state_in.particle_q,
                                  state_in.particle_qd,
                                  model.particle_inv_mass,
                                  model.spring_indices,
                                  model.spring_rest_length,
                                  model.spring_stiffness,
                                  model.spring_damping,
                                  dt
                              ],
                              outputs=[state_out.particle_f],
                              device=model.device)

                # tetrahedral FEM
                if (model.tet_count):

                    wp.launch(kernel=solve_tetrahedra,
                              dim=model.tet_count,
                              inputs=[
                                  particle_q,
                                  particle_qd,
                                  model.particle_inv_mass,
                                  model.tet_indices,
                                  model.tet_poses,
                                  model.tet_activations,
                                  model.tet_materials,
                                  dt,
                                  self.soft_body_relaxation
                              ],
                              outputs=[state_out.particle_f],
                              device=model.device)

                # apply updates
                wp.launch(kernel=apply_deltas,
                          dim=model.particle_count,
                          inputs=[state_in.particle_q,
                                  state_in.particle_qd,
                                  particle_q,
                                  state_out.particle_f,
                                  dt],
                          outputs=[particle_q,
                                   particle_qd],
                          device=model.device)

            # rigid bodies
            # ----------------------------

            if (model.body_count):
                state_out.body_f.zero_()
                state_out.body_q_prev.assign(state_in.body_q)
                state_out.body_qd_prev.assign(state_in.body_qd)

                wp.launch(
                    kernel=apply_joint_torques,
                    dim=model.joint_count,
                    inputs=[
                        state_in.body_q,
                        model.body_com,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_axis,
                        model.joint_act,
                    ],
                    outputs=[
                        state_out.body_f
                    ],
                    device=model.device)

                wp.launch(
                    kernel=integrate_bodies,
                    dim=model.body_count,
                    inputs=[
                        state_in.body_q,
                        state_in.body_qd,
                        state_out.body_f,
                        model.body_com,
                        model.body_mass,
                        model.body_inertia,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.gravity,
                        self.angular_damping,
                        dt,
                    ],
                    outputs=[
                        state_out.body_q,
                        state_out.body_qd
                    ],
                    device=model.device)


                # -------------------------------------
                # integrate bodies

                for i in range(self.iterations):
                    # print(f"### iteration {i} / {self.iterations-1}")
                    # state_out.body_deltas.zero_()

                    if (model.joint_count):
                        wp.launch(kernel=solve_body_joints,
                                dim=model.joint_count,
                                inputs=[
                                    state_out.body_q,
                                    state_out.body_qd,
                                    model.body_com,
                                    model.body_inv_mass,
                                    model.body_inv_inertia,
                                    model.joint_q_start,
                                    model.joint_qd_start,
                                    model.joint_type,
                                    model.joint_parent,
                                    model.joint_child,
                                    model.joint_X_p,
                                    model.joint_X_c,
                                    model.joint_axis,
                                    model.joint_target,
                                    model.joint_target_ke,
                                    model.joint_target_kd,
                                    model.joint_limit_lower,
                                    model.joint_limit_upper,
                                    model.joint_twist_lower,
                                    model.joint_twist_upper,
                                    model.joint_linear_compliance,
                                    model.joint_angular_compliance,
                                    self.joint_angular_relaxation,
                                    self.joint_positional_relaxation,
                                    dt
                                ],
                                outputs=[
                                    state_out.body_deltas
                                ],
                                device=model.device)
                        
                        # apply updates
                        wp.launch(kernel=apply_body_deltas,
                                dim=model.body_count,
                                inputs=[
                                    state_out.body_q,
                                    state_out.body_qd,
                                    model.body_com,
                                    model.body_inertia,
                                    model.body_inv_mass,
                                    model.body_inv_inertia,
                                    state_out.body_deltas,
                                    dt
                                ],
                                outputs=[
                                    state_out.body_q,
                                    state_out.body_qd,
                                ],
                                device=model.device)

                    # Solve rigid contact constraints
                    if (model.rigid_contact_max and model.body_count):
                        model.rigid_contact_inv_weight.zero_()
                        model.rigid_active_contact_distance.zero_()

                        wp.launch(kernel=update_body_contact_weights,
                            dim=model.rigid_contact_max,
                            inputs=[
                                state_out.body_q,
                                model.rigid_contact_count,
                                model.rigid_contact_body0,
                                model.rigid_contact_body1,
                                model.rigid_contact_point0,
                                model.rigid_contact_point1,
                                model.rigid_contact_normal,
                                model.rigid_contact_thickness,
                                model.rigid_contact_shape0,
                                model.rigid_contact_shape1,
                                model.shape_transform
                            ],
                            outputs=[
                                model.rigid_contact_inv_weight,
                                model.rigid_active_contact_point0,
                                model.rigid_active_contact_point1,
                                model.rigid_active_contact_distance,
                            ],
                            device=model.device)

                        if (i == 0):
                            # remember the contacts from the first iteration
                            model.rigid_active_contact_distance_prev.assign(model.rigid_active_contact_distance)

                        if (not self.contact_con_weighting):
                            model.rigid_contact_inv_weight.fill_(1.0)

                        wp.launch(kernel=solve_body_contact_positions,
                            dim=model.rigid_contact_max,
                            inputs=[
                                state_out.body_q,
                                state_out.body_qd,
                                model.body_com,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.rigid_contact_count,
                                model.rigid_contact_body0,
                                model.rigid_contact_body1,
                                model.rigid_contact_offset0,
                                model.rigid_contact_offset1,
                                model.rigid_contact_normal,
                                model.rigid_contact_shape0,
                                model.rigid_contact_shape1,
                                model.rigid_active_contact_point0,
                                model.rigid_active_contact_point1,
                                model.rigid_active_contact_distance,
                                model.shape_materials,
                                model.rigid_contact_inv_weight,
                                self.contact_normal_relaxation,
                                dt,
                                self.max_rigid_contact_penetration,
                                model.rigid_contact_torsional_friction,
                                model.rigid_contact_rolling_friction,
                            ],
                            outputs=[
                                state_out.body_deltas,
                            ],
                            device=model.device)

                    # apply updates
                    wp.launch(kernel=apply_body_deltas,
                            dim=model.body_count,
                            inputs=[
                                state_out.body_q,
                                state_out.body_qd,
                                model.body_com,
                                model.body_inertia,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                state_out.body_deltas,
                                dt
                            ],
                            outputs=[
                                state_out.body_q,
                                state_out.body_qd,
                            ],
                            device=model.device)

                # update body velocities
                wp.launch(kernel=update_body_velocities,
                        dim=model.body_count,
                        inputs=[
                            state_out.body_q,
                            state_out.body_q_prev,
                            model.body_com,
                            dt
                        ],
                        outputs=[
                            state_out.body_qd
                        ],
                        device=model.device)

                if (model.has_restitution):
                    wp.launch(kernel=apply_rigid_restitution,
                            dim=model.rigid_contact_max,
                            inputs=[
                                state_out.body_q,
                                state_out.body_qd,
                                state_out.body_q_prev,
                                state_out.body_qd_prev,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.rigid_contact_count,
                                model.rigid_contact_body0,
                                model.rigid_contact_body1,
                                model.rigid_contact_point0,
                                model.rigid_contact_point1,
                                model.rigid_contact_offset0,
                                model.rigid_contact_offset1,
                                model.rigid_contact_normal,
                                model.rigid_contact_shape0,
                                model.rigid_contact_shape1,
                                model.shape_materials,
                                model.rigid_active_contact_distance_prev,
                                model.rigid_contact_inv_weight,
                                model.gravity,
                                model.rigid_contact_bounce_threshold,
                                dt,
                            ],
                            outputs=[
                                state_out.body_deltas,
                            ],
                            device=model.device)

                    wp.launch(kernel=apply_body_delta_velocities,
                            dim=model.body_count,
                            inputs=[
                                state_out.body_qd,
                                state_out.body_deltas,
                            ],
                            outputs=[
                                state_out.body_qd
                            ],
                            device=model.device)

            state_out.particle_q = particle_q
            state_out.particle_qd = particle_qd

            return state_out
