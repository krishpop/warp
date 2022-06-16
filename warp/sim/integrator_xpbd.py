# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp


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
def integrate_bodies(body_q: wp.array(dtype=wp.transform),
                     body_qd: wp.array(dtype=wp.spatial_vector),
                     body_f: wp.array(dtype=wp.spatial_vector),
                     body_com: wp.array(dtype=wp.vec3),
                     m: wp.array(dtype=float),
                     I: wp.array(dtype=wp.mat33),
                     inv_m: wp.array(dtype=float),
                     inv_I: wp.array(dtype=wp.mat33),
                     gravity: wp.vec3,
                     dt: float,
                     body_q_new: wp.array(dtype=wp.transform),
                     body_qd_new: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    f = body_f[tid]

    # masses
    mass = m[tid]
    inv_mass = inv_m[tid]     # 1 / mass

    inertia = I[tid]
    inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix

    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)

    # unpack spatial wrench
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    x_com = x0 + wp.quat_rotate(r0, body_com[tid])

    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt
    x1 = x_com + v1 * dt

    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb,
                                               inertia*wb)   # coriolis forces

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping, todo: expose
    # w1 = w1*(1.0-0.1*dt)

    body_q_new[tid] = wp.transform(x1 - wp.quat_rotate(r1, body_com[tid]), r1)
    body_qd_new[tid] = wp.spatial_vector(w1, v1)


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

    l = length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = dot(dir, vij)

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
    r_s = wp.sqrt(dot(f1, f1) + dot(f2, f2) + dot(f3, f3))
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

    grad1 = vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    grad2 = vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    grad3 = vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    denom = dot(grad0, grad0)*w0 + dot(grad1, grad1)*w1 + \
        dot(grad2, grad2)*w2 + dot(grad3, grad3)*w3
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

    denom = dot(grad0, grad0)*w0 + dot(grad1, grad1)*w1 + \
        dot(grad2, grad2)*w2 + dot(grad3, grad3)*w3
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

    # clear forces
    delta[tid] = wp.vec3()


@wp.kernel
def eval_body_contacts(body_q: wp.array(dtype=wp.transform),
                       body_qd: wp.array(dtype=wp.spatial_vector),
                       body_com: wp.array(dtype=wp.vec3),
                       contact_body: wp.array(dtype=int),
                       contact_point: wp.array(dtype=wp.vec3),
                       contact_dist: wp.array(dtype=float),
                       contact_mat: wp.array(dtype=int),
                       materials: wp.array(dtype=wp.vec4),
                       body_f: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    c_body = contact_body[tid]
    c_point = contact_point[tid]
    c_dist = contact_dist[tid]
    c_mat = contact_mat[tid]

    X_wb = body_q[c_body]
    v_wc = body_qd[c_body]

    # unpack spatial twist
    w = wp.spatial_top(v_wc)
    v = wp.spatial_bottom(v_wc)

    n = vec3(0.0, 1.0, 0.0)

    # transform point to world space
    # add on 'thickness' of shape, e.g.: radius of sphere/capsule
    cp = wp.transform_point(X_wb, c_point) - n * c_dist

    # moment arm around center of mass
    r = cp - wp.transform_point(X_wb, body_com[c_body])

    # contact point velocity
    dpdt = v + wp.cross(w, r)

    # check ground contact
    c = wp.dot(n, cp)

    if (c > 0.0):
        return

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    mat = materials[c_mat]

    ke = mat[0]       # restitution coefficient
    kd = mat[1]       # damping coefficient
    kf = mat[2]       # friction coefficient
    mu = mat[3]       # coulomb friction

    vn = wp.dot(n, dpdt)
    vt = dpdt - n * vn

    # normal force
    fn = c * ke

    # damping force
    # again, velocity into the ground, negative
    fd = wp.min(vn, 0.0) * kd * wp.step(c)

    # viscous friction
    #ft = vt*kf

    # # Coulomb friction (box)
    # lower = mu * (fn + fd)   # negative
    # upper = 0.0 - lower      # positive, workaround for no unary ops

    # vx = wp.clamp(wp.dot(vec3(kf, 0.0, 0.0), vt), lower, upper)
    # vz = wp.clamp(wp.dot(vec3(0.0, 0.0, kf), vt), lower, upper)

    # ft = wp.vec3(vx, 0.0, vz) * wp.step(c)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    ft = wp.normalize(vt)*wp.min(kf*wp.length(vt), 0.0 - mu*(fn + fd))

    f_total = n * (fn + fd) + ft
    t_total = wp.cross(r, f_total)

    wp.atomic_sub(body_f, c_body, wp.spatial_vector(t_total, f_total))


@wp.func
def world_to_body(v: wp.vec3, pose: wp.transform) -> wp.vec3:
    return v
    # Project a vector from world frame into rest frame of body
    # q = wp.transform_get_rotation(pose)
    # projected = wp.quat_rotate_inv(q, v)
    # return projected

@wp.func
def body_to_world(v: wp.vec3, pose: wp.transform) -> wp.vec3:
    return v
    # Project a vector from world frame into rest frame of body
    # q = wp.transform_get_rotation(pose)
    # projected = wp.quat_rotate(q, v)
    # return projected

@wp.func
def get_inverse_mass(
    tf: wp.transform,
    normal: wp.vec3,
    arm: wp.vec3,
    I_inv: wp.mat33,
    m_inv: float,
    angular: bool,
) -> float:
    # Compute the inverse mass of a point relative to the center of mass
    # of a rigid body.
    if m_inv == 0.0:
        return m_inv
    q = wp.transform_get_rotation(tf)
    p = wp.transform_get_translation(tf)
    
    if angular:
        rn = wp.quat_rotate_inv(q, normal)
    else:
        rn = wp.quat_rotate_inv(q, wp.cross(arm-p, normal))


@wp.func
def positional_correction(
    dx: wp.vec3,
    tf1: wp.transform,
    tf2: wp.transform,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    alpha_tilde: float,
    # lambda_prev: float,
    deltas: wp.array(dtype=wp.spatial_vector),
    body_1: int,
    body_2: int,
) -> wp.vec3:
    # Computes the correction impulse for a positional constraint.
    # The impulse is applied the two bodies if their respective indices are >= 0.

    # Compliance corresponds to linear compliance alpha / dt^2 (corrected for timestep)
    # Equation 2-5 of [1]
    # [1] https://matthias-research.github.io/pages/publications/PBDBodies.pdf

    c = wp.length(dx)
    n = wp.normalize(dx)

    # project variables to body rest frame as they are in local matrix.
    n_local_1 = world_to_body(n, tf1)
    n_local_2 = world_to_body(n, tf2)
    r1 = wp.transform_get_translation(tf1)
    r2 = wp.transform_get_translation(tf2)
    r1_local = world_to_body(r1, tf1)
    r2_local = world_to_body(r2, tf2)

    # Eq. 1-2
    r1xn = wp.cross(r1_local, n_local_1)
    r2xn = wp.cross(r2_local, n_local_2)
    w1 = m_inv1 + wp.dot(r1xn, I_inv1 * r1xn)
    w2 = m_inv2 + wp.dot(r2xn, I_inv2 * r2xn)

    lambda_prev = 0.0
    d_lambda = (-c - alpha_tilde * lambda_prev) / (w1 + w2 + alpha_tilde)
    # TODO consider lambda_prev?
    p = d_lambda * n

    if body_1 >= 0 and m_inv1 > 0.0:
        dp = body_to_world(p, tf1)
        # dp = p * m_inv1
        dq = wp.cross(r1_local, p)
        wp.atomic_sub(deltas, body_1, wp.spatial_vector(dq, dp))
    if body_2 >= 0 and m_inv2 > 0.0:
        dp = body_to_world(p, tf2)
        # dp = p * m_inv2
        dq = wp.cross(r2_local, p)
        wp.atomic_add(deltas, body_2, wp.spatial_vector(dq, dp))
    return p

@wp.func
def angular_correction(
    dq: wp.vec3,
    tf1: wp.transform,
    tf2: wp.transform,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    alpha_tilde: float,
    # lambda_prev: float,
    deltas: wp.array(dtype=wp.spatial_vector),
    body_1: int,
    body_2: int,
) -> wp.vec3:
    # Computes the correction impulse for an angular constraint.
    # The impulse is applied the two bodies if their respective indices are >= 0.

    # Compliance corresponds to linear compliance alpha / dt^2 (corrected for timestep)
    # Equation 2-5 of [1]
    # [1] https://matthias-research.github.io/pages/publications/PBDBodies.pdf

    theta = wp.length(dq)
    n = wp.normalize(dq)

    # project variables to body rest frame as they are in local matrix.
    n_local_1 = world_to_body(n, tf1)
    n_local_2 = world_to_body(n, tf2)

    # Eq. 11-12
    w1 = wp.dot(n_local_1, I_inv1 * n_local_1)
    w2 = wp.dot(n_local_2, I_inv2 * n_local_2)

    lambda_prev = 0.0
    d_lambda = (-theta - alpha_tilde * lambda_prev) / (w1 + w2 + alpha_tilde)
    # TODO consider lambda_prev?
    p = d_lambda * n

    dp = wp.vec3(0.0)
    if body_1 >= 0 and m_inv1 > 0.0:
        # pose1 = poses[body_1]
        # x1 = wp.transform_get_translation(pose1)
        # q1 = wp.transform_get_rotation(pose1)
        dq = p
        wp.atomic_sub(deltas, body_1, wp.spatial_vector(dq, dp))
        # dq = wp.quat(I_inv1 * p, 0.0)
        # q1 += wp.quat_multiply(0.5 * dq, q1)
        # q1 = wp.quat_normalize(q1)
        # poses[body_1] = wp.transform(x1, q1)
    if body_2 >= 0 and m_inv2 > 0.0:
        # pose2 = poses[body_2]
        # x2 = wp.transform_get_translation(pose2)
        # q2 = wp.transform_get_rotation(pose2)
        dq = p
        wp.atomic_add(deltas, body_2, wp.spatial_vector(dq, dp))
        # dq = wp.quat(I_inv2 * p, 0.0)
        # q2 += wp.quat_multiply(0.5 * dq, q2)
        # q2 = wp.quat_normalize(q2)
        # poses[body_2] = wp.transform(x2, q2)
    return p

@wp.kernel
def apply_body_deltas(
    poses_in: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_I: wp.array(dtype=wp.mat33),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),
    deltas: wp.array(dtype=wp.spatial_vector),
    poses_out: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    inv_m = body_inv_m[tid]
    if inv_m == 0.0:
        return
    tf = poses_in[tid]
    delta = deltas[tid]
    com = body_com[tid]

    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)

    # update position
    dp = wp.spatial_bottom(delta)
    epsilon = 1.0 # e-3
    p += dp * inv_m * epsilon

    # update orientation
    I_inv = body_inv_I[tid]
    dq = I_inv * wp.spatial_top(delta)
    # TODO reactivate
    q += 0.5 * wp.quat(dq, 0.0) * q * epsilon
    q = wp.normalize(q)

    poses_out[tid] = wp.transform(p, q)

# @wp.func
# def correct_velocity(
#     dv: wp.vec3,
#     r1: wp.vec3,
#     r2: wp.vec3,
#     body1_q: wp.transform,
#     body2_q: wp.transform,
#     body1_m_inv: float,
#     body2_m_inv: float,
#     body1_I_inv: wp.mat33,
#     body2_I_inv: wp.mat33,
#     alpha_tilde: float,
#     # lambda_prev: float,
#     poses: wp.array(dtype=wp.transform),
#     body_1: int,
#     body_2: int,
# ) -> wp.vec3:
#     p = dv / ()

# @wp.func
# def xpbd_angular_impluse(
#     dq: wp.vec3,
#     body1_q: wp.transform,
#     body2_q: wp.transform,
#     body1_I_inv: wp.mat33,
#     body2_I_inv: wp.mat33,
#     compliance: float,
# ) -> wp.vec3:
#     # Calculates the lagrange angular multiplier for two bodies.
#     # Compliance corresponds to alpha / dt^2 (corrected for timestep)
#     # Equation 11-16 of [1].
#     # [1] https://matthias-research.github.io/pages/publications/PBDBodies.pdf

#     theta = wp.length(dq)
#     n = wp.normalize(dq)

#     # project variables to body rest frame as they are in local matrix.
#     n_local_1 = world_to_body(n, body1_q)
#     n_local_2 = world_to_body(n, body2_q)

#     w1 = wp.dot(n_local_1, body1_I_inv * n_local_1)
#     w2 = wp.dot(n_local_2, body2_I_inv * n_local_2)

#     d_lambda = -theta / (w1 + w2 + compliance)
#     p = d_lambda * n

#     return p

@wp.func
def quat_basis_vector_a(q: wp.quat) -> wp.vec3:
    x2 = q[0] * 2.0
    w2 = q[3] * 2.0
    return wp.vec3((q[3] * w2) - 1.0 + q[0] * x2, (q[2] * w2) + q[1] * x2, (-q[1] * w2) + q[2] * x2)

@wp.func
def quat_basis_vector_b(q: wp.quat) -> wp.vec3:
    y2 = q[1] * 2.0
    w2 = q[3] * 2.0
    return wp.vec3((-q[2] * w2) + q[0] * y2, (q[3] * w2) - 1.0 + q[1] * y2, (q[0] * w2) + q[2] * y2)

@wp.func
def quat_basis_vector_c(q: wp.quat) -> wp.vec3:
    z2 = q[2] * 2.0
    w2 = q[3] * 2.0
    return wp.vec3((q[1] * w2) + q[0] * z2, (-q[0] * w2) + q[1] * z2, (q[3] * w2) - 1.0 + q[2] * z2)

@wp.kernel
def solve_body_joints(body_q: wp.array(dtype=wp.transform),
                      body_qd: wp.array(dtype=wp.spatial_vector),
                      body_com: wp.array(dtype=wp.vec3),
                      body_mass: wp.array(dtype=float),
                      body_I: wp.array(dtype=wp.mat33),
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
                      joint_act: wp.array(dtype=float),
                      joint_target_ke: wp.array(dtype=float),
                      joint_target_kd: wp.array(dtype=float),
                      joint_limit_lower: wp.array(dtype=float),
                      joint_limit_upper: wp.array(dtype=float),
                      joint_twist_lower: wp.array(dtype=float),
                      joint_twist_upper: wp.array(dtype=float),
                      joint_linear_compliance: wp.array(dtype=float),
                      joint_angular_compliance: wp.array(dtype=float),
                      dt: float,
                      deltas: wp.array(dtype=wp.spatial_vector)):
    tid = wp.tid()
    type = joint_type[tid]

    if (type == wp.sim.JOINT_FREE):
        return

    # print("joint")
    # print(type)

    # rigid body indices of the child and parent
    b1 = joint_child[tid]
    b2 = joint_parent[tid]
    # b1 = joint_parent[tid]
    # b2 = joint_child[tid]

    tf1 = joint_X_c[tid]
    m_inv1 = 0.0
    I_inv1 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if b1 >= 0:
        tf1 = body_q[b1] * tf1
        m_inv1 = body_inv_m[b1]
        I_inv1 = body_inv_I[b1]
    tf2 = joint_X_p[tid]
    m_inv2 = 0.0
    I_inv2 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if b2 >= 0:
        tf2 = body_q[b2] * tf2
        m_inv2 = body_inv_m[b2]
        I_inv2 = body_inv_I[b2]

    # print("tf1")
    # print(b1)
    # print(tf1)
    # print("tf2")
    # print(b2)
    # print(tf2)

    # joint properties (for 1D joints)
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    axis = joint_axis[tid]

    if (type == wp.sim.JOINT_FIXED):
        limit_lower = 0.0
        limit_upper = 0.0
    else:
        limit_lower = joint_limit_lower[q_start]
        limit_upper = joint_limit_upper[q_start]

    # transform the joint axis to the local frame of the child
    # axis1 = wp.transform_vector(tf1, axis)

    linear_alpha = joint_linear_compliance[tid]
    angular_alpha = joint_angular_compliance[tid]

    linear_alpha_tilde = linear_alpha / dt / dt
    angular_alpha_tilde = angular_alpha / dt / dt

    # prevent division by zero
    linear_alpha_tilde = wp.max(linear_alpha_tilde, 1e-6)
    angular_alpha_tilde = wp.max(angular_alpha_tilde, 1e-6)

    # handle angular constraints
    q1 = wp.transform_get_rotation(tf1)
    q2 = wp.transform_get_rotation(tf2)
    if (type == wp.sim.JOINT_REVOLUTE):
        # align joint axes
        a1 = quat_basis_vector_a(q1)
        a2 = quat_basis_vector_a(q2)
        # Eq. 20
        corr = wp.cross(a1, a2)
        angular_correction(
            corr, tf1, tf2, m_inv1, m_inv2, I_inv1, I_inv2,
            angular_alpha_tilde, deltas, b1, b2)
    if (type == wp.sim.JOINT_FIXED):
        # Eq. 18-19
        q = q1 * wp.quat_inverse(q2)
        corr = 2.0 * wp.vec3(q[0], q[1], q[2])
        angular_correction(
            corr, tf1, tf2, m_inv1, m_inv2, I_inv1, I_inv2,
            angular_alpha_tilde, deltas, b1, b2)
    if (type == wp.sim.JOINT_BALL):
        # TODO: implement Eq. 22-24
        wp.print("Ball joint not implemented")
        return

    # handle positional constraints
    r1 = wp.transform_get_translation(tf1)
    r2 = wp.transform_get_translation(tf2)
    dr = r2 - r1
    dx = dr  # TODO add support for separation distance? (Eq. 25)

    lower_pos_limits = wp.vec3(0.0)
    upper_pos_limits = wp.vec3(0.0)
    if (type == wp.sim.JOINT_PRISMATIC):
        lower_pos_limits = axis * limit_lower
        upper_pos_limits = axis * limit_upper

    corr = wp.vec3(0.0)

    normal_a = wp.normalize(quat_basis_vector_a(q1))
    # print("unit x")
    # print(normal_a)
    d = wp.dot(normal_a, dx)
    if (d < lower_pos_limits[0]):
        corr -= normal_a * (lower_pos_limits[0] - d)
    if (d > upper_pos_limits[0]):
        # TODO which one?
        # corr -= normal_a * (d - upper_pos_limits[0])
        corr -= normal_a * (upper_pos_limits[0] - d)  # from Matthias
    normal_b = wp.normalize(quat_basis_vector_b(q1))
    # print("unit y")
    # print(normal_b)
    d = wp.dot(normal_b, dx)
    if (d < lower_pos_limits[1]):
        corr -= normal_b * (lower_pos_limits[1] - d)
    if (d > upper_pos_limits[1]):
        # TODO which one?
        # corr -= normal_b * (d - upper_pos_limits[1])
        corr -= normal_b * (upper_pos_limits[1] - d)  # from Matthias
    normal_c = wp.normalize(quat_basis_vector_c(q1))
    # print("unit z")
    # print(normal_c)
    d = wp.dot(normal_c, dx)
    if (d < lower_pos_limits[2]):
        corr -= normal_c * (lower_pos_limits[2] - d)
    if (d > upper_pos_limits[2]):
        # TODO which one?
        # corr -= normal_c * (d - upper_pos_limits[2])
        corr -= normal_c * (upper_pos_limits[2] - d)  # from Matthias
    
    positional_correction(corr, tf1, tf2, m_inv1, m_inv2, I_inv1, I_inv2,
                          linear_alpha_tilde, deltas, b1, b2)

    # if (tid > 0):
    #     print("corr")
    #     print(corr)



# @wp.kernel
def solve_body_joints2(body_q: wp.array(dtype=wp.transform),
                      body_qd: wp.array(dtype=wp.spatial_vector),
                      body_com: wp.array(dtype=wp.vec3),
                      body_mass: wp.array(dtype=float),
                      body_I: wp.array(dtype=wp.mat33),
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
                      joint_act: wp.array(dtype=float),
                      joint_target_ke: wp.array(dtype=float),
                      joint_target_kd: wp.array(dtype=float),
                      joint_limit_lower: wp.array(dtype=float),
                      joint_limit_upper: wp.array(dtype=float),
                      dt: float,
                      deltas: wp.array(dtype=wp.spatial_vector)):

    # tid in this kernel is *joint* index
    tid = wp.tid()

    # rigid body indices of the child and parent
    c_child = joint_child[tid]
    c_parent = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    r_p = wp.vec3()
    w_p = wp.vec3()  # angular velocity of parent
    v_p = wp.vec3()  # linear velocity of parent

    if (c_parent >= 0):  # (joint with no parent has index -1)
        X_wp = body_q[c_parent]*X_wp  # parent transform
        # moment arm from parent COM to joint pos
        r_p = wp.transform_get_translation(
            X_wp) - wp.transform_point(body_q[c_parent], body_com[c_parent])

        twist_p = body_qd[c_parent]

        w_p = wp.spatial_top(twist_p)  # angular velocity of parent
        # linear velocity of parent (at point of joint!)
        v_p = wp.spatial_bottom(twist_p) + wp.cross(w_p, r_p)

    # child transform and moment arm (NB. X_cj is identity in WARP importer)
    X_wc = body_q[c_child]  # *X_cj
    r_c = wp.transform_get_translation(
        X_wc) - wp.transform_point(body_q[c_child], body_com[c_child])

    twist_c = body_qd[c_child]

    w_c = wp.spatial_top(twist_c)  # angular velocity of child
    v_c = wp.spatial_bottom(twist_c) + wp.cross(w_c,
                                                r_c)

    # joint properties (for 1D joints)
    # indexed by body indices of child (with joint associated connecting to parent)
    q_start = joint_q_start[c_child]
    qd_start = joint_qd_start[c_child]
    type = joint_type[c_child]
    axis = joint_axis[c_child]

    target = joint_target[qd_start]
    target_ke = joint_target_ke[qd_start]  # stiffness
    target_kd = joint_target_kd[qd_start]  # damping

    limit_lower = joint_limit_lower[qd_start]
    limit_upper = joint_limit_upper[qd_start]

    act = joint_act[qd_start]

    m_p = body_mass[c_parent]
    m_c = body_mass[c_child]

    inv_m_p = body_inv_m[c_parent]
    inv_m_c = body_inv_m[c_child]

    # NB - in body's local rest frame
    I_p = body_I[c_parent]
    I_c = body_I[c_child]

    inv_I_p = body_inv_I[c_parent]
    inv_I_c = body_inv_I[c_child]

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # translational error
    x_err = x_c - x_p
    r_err = wp.quat_inverse(q_p)*q_c
    # v_err = v_c - v_p
    # w_err = w_c - w_p

    # # reduce angular damping stiffness for stability
    # angular_damping_scale = 0.01

    if (type == wp.sim.JOINT_FREE):
        return



    p_pos = wp.vec3()
    p_rot = wp.vec3()

    if type == wp.sim.JOINT_FIXED:

        p_pos = xpbd_positional_impulse(
            x_err, r_p, r_c, body_q[c_parent], body_q[c_child], inv_m_p, inv_m_c, inv_I_p, inv_I_c, a_hat
        )

        # equation 19

        q_err = 2. * wp.vec3(r_err[0], r_err[1], r_err[2])

        p_rot = xpbd_angular_impluse(
            q_err, body_q[c_parent], body_q[c_child], inv_I_p, inv_I_c, a_hat
        )

    p_pos_parent = world_to_body(p_pos, body_q[c_parent])
    r_pos_parent = world_to_body(r_p, body_q[c_parent])
    p_rot_parent = world_to_body(p_rot, body_q[c_parent])

    # deltas appear as [ang_delta, lin_delta] (like velocities)
    # equations 6-9 and 15-16 of apaper
    delta_parent = wp.spatial_vector(
        -0.5 * (inv_I_p * (
            wp.cross(r_pos_parent, p_pos_parent)
            + p_rot_parent
        )),
        -p_pos * inv_m_p,
    )

    # project vectors to rest reference frame so that they project onto inertia matrix
    p_pos_child = world_to_body(p_pos, body_q[c_child])
    r_pos_child = world_to_body(r_c, body_q[c_child])
    p_rot_child = world_to_body(p_rot, body_q[c_child])

    # deltas appear as [ang_delta, lin_delta] (like velocities)
    delta_child = wp.spatial_vector(
        0.5 * (inv_I_c * (
            wp.cross(r_pos_child, p_pos_child)
            + p_rot_child
        )),
        p_pos * inv_m_c,
    )

    wp.atomic_add(deltas, c_child, delta_child)
    wp.atomic_add(deltas, c_parent, delta_parent)


@wp.kernel
def update_body_velocities(
    poses: wp.array(dtype=wp.transform),
    poses_prev: wp.array(dtype=wp.transform),
    # deltas: wp.array(dtype=wp.spatial_vector),
    dt: float,
    # body_q_out: wp.array(dtype=wp.transform),
    vel_qd_out: wp.array(dtype=wp.spatial_vector)
):

    tid = wp.tid()

    pose = poses[tid]
    pose_prev = poses_prev[tid]

    x = wp.transform_get_translation(pose)
    x_prev = wp.transform_get_translation(pose_prev)

    q = wp.transform_get_rotation(pose)
    q_prev = wp.transform_get_rotation(pose_prev)

    v = (x - x_prev) / dt
    dq = q * wp.quat_inverse(q_prev)  #XXX  wp.quat_inverse(q_prev) * q ?

    omega = 2.0/dt * wp.vec3(dq[0], dq[1], dq[2])
    if dq[3] < 0.0:
        omega = -omega

    vel_qd_out[tid] = wp.spatial_vector(omega, v)

    # weight = 1.0f/Max(1.0f, linearDeltas[tid].w);

    # weight = 1.0
    # # if inv_mass > 0.0:
    # #     weight = wp.min(1.0, inv_mass)

    # # constraint deltas
    # d = deltas[tid]

    # q_prev = body_q[tid]
    # q_pred = body_q_pred[tid]

    # # position and orientation before integration
    # pos_prev = wp.transform_get_translation(q_prev)
    # theta_prev = wp.transform_get_rotation(q_prev)

    # # position and orientation after integration
    # pos_pred = wp.transform_get_translation(q_pred)
    # theta_pred = wp.transform_get_rotation(q_pred)

    # # pos_delta = wp.spatial_bottom(d) * weight * dt * dt
    # # ang_delta = wp.quat(wp.spatial_bottom(d), 0.0) * theta * 0.5 * weight * dt * dt
    # # delta vectors are [ang_delta, pos_delta]
    # pos_delta = wp.spatial_bottom(d) * weight  # * weight * dt * dt
    # # theta_prev # * theta * 0.5 * weight * dt * dt
    # ang_delta = wp.quat(wp.spatial_top(d), 0.0) * theta_pred * weight

    # new_pos = pos_pred + pos_delta
    # new_rot = wp.normalize(theta_pred + ang_delta)  # theta_pred is unstable

    # # body_qd_out[tid] = body_qd[tid] + d * dt * weight

    # new_linvel = (new_pos - pos_prev) / dt
    # delta_rot = new_rot * wp.quat_inverse(theta_prev)
    # new_angvel = 2. * wp.vec3(delta_rot[0], delta_rot[1], delta_rot[2]) / dt

    # #new_angvel = 2. * wp.vec3()/ dt

    # if delta_rot[3] < 0:
    #     new_angvel = - new_angvel

    # body_q_out[tid] = wp.transform(new_pos, new_rot)
    # # velocities are [w, v] (angular velocity on top)
    # body_qd_out[tid] = wp.spatial_vector(new_angvel, new_linvel)

    # # (Vec3&)rigids[tid].linearVel += Vec3(linearDeltas[tid])*dt*weight;
    # # (Vec3&)rigids[tid].angularVel += Vec3(angularDeltas[tid])*dt*weight;

    # # clear rigid deltas for next iteration
    # deltas[tid] = wp.spatial_vector()


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

    def __init__(self, iterations, relaxation):

        self.iterations = iterations
        self.relaxation = relaxation

    def simulate(self, model, state_in, state_out, dt):

        with wp.ScopedTimer("simulate", False):

            particle_q = None
            particle_qd = None

            if (model.particle_count):
                particle_q = wp.zeros_like(state_in.particle_q)
                particle_qd = wp.zeros_like(state_in.particle_qd)

                # alloc particle force buffer
                state_out.particle_f.zero_()

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
                                  self.relaxation
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

            # alloc rigid body force buffer
            if (model.body_count):
                state_out.body_f.zero_()
                # state_out.body_deltas.zero_()

            # integrate rigid bodies
            if (model.body_count):
                wp.launch(
                    kernel=integrate_bodies,
                    dim=model.body_count,
                    inputs=[
                        state_in.body_q,
                        state_in.body_qd,
                        state_in.body_f,
                        model.body_com,
                        model.body_mass,
                        model.body_inertia,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.gravity,
                        dt,
                    ],
                    outputs=[
                        state_out.body_q,
                        state_out.body_qd
                    ],
                    device=model.device)

            # body_f = None

            if state_in.body_count:
                # body_q_pred = wp.zeros_like(state_in.body_q)
                # body_qd_pred = wp.zeros_like(state_in.body_qd)
                new_body_q = wp.zeros_like(state_in.body_q)
                # body_f = state_in.body_f
                # body_deltas = state_out.body_deltas
                state_out.body_deltas.zero_()

                # body_deltas = wp.zeros_like(state_in.body_q)

                #compute_forces(model, state_in, particle_f, body_f)

                # if (model.body_count and model.contact_count > 0 and model.ground):

                #     wp.launch(kernel=eval_body_contacts,
                #                 dim=model.contact_count,
                #                 inputs=[
                #                     state_in.body_q,
                #                     state_in.body_qd,
                #                     model.body_com,
                #                     model.contact_body0,
                #                     model.contact_point0,
                #                     model.contact_dist,
                #                     model.contact_material,
                #                     model.shape_materials
                #                 ],
                #                 outputs=[
                #                     body_f
                #                 ],
                #                 device=model.device)

                # -------------------------------------
                # integrate bodies

                new_body_q = wp.clone(state_out.body_q)

                for i in range(self.iterations):

                    wp.launch(kernel=solve_body_joints,
                              dim=model.joint_count,
                              inputs=[
                                  new_body_q,
                                  state_out.body_qd,
                                  model.body_com,
                                  model.body_mass,
                                  model.body_inertia,
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
                                  model.joint_act,
                                  model.joint_target_ke,
                                  model.joint_target_kd,
                                  model.joint_limit_lower,
                                  model.joint_limit_upper,
                                  model.joint_twist_lower,
                                  model.joint_twist_upper,
                                  model.joint_linear_compliance,
                                  model.joint_angular_compliance,
                                  dt / self.iterations
                              ],
                              outputs=[
                                  state_out.body_deltas
                              ],
                              device=model.device)

                    # apply updates
                    wp.launch(kernel=apply_body_deltas,
                            dim=model.body_count,
                            inputs=[
                                new_body_q,
                                model.body_com,
                                model.body_mass,
                                model.body_inertia,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                state_out.body_deltas
                            ],
                            outputs=[
                                new_body_q,
                            ],
                            device=model.device)

                # update body velocities
                wp.launch(kernel=update_body_velocities,
                        dim=model.body_count,
                        inputs=[
                            new_body_q,
                            state_in.body_q,
                            dt
                        ],
                        outputs=[
                            state_out.body_qd
                        ],
                        device=model.device)

                state_out.body_q = new_body_q

            state_out.particle_q = particle_q
            state_out.particle_qd = particle_qd

            # state_out.body_q = body_q_pred
            # state_out.body_qd = body_qd_pred

            return state_out
