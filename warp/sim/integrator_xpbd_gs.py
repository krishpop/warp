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

    # x1 = x0 + v1 * dt

    # angular part
    # wb = wp.quat_rotate_inv(r0, w0)
    # gyr = -(inv_inertia * wp.cross(wb, inertia*wb))
    # w1 = w0 + dt * wp.quat_rotate(r0, gyr)

    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb,
                                               inertia*wb)   # coriolis forces

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping, todo: expose
    # w1 = w1*(1.0-0.1*dt)

    body_q_new[tid] = wp.transform(x1 - wp.quat_rotate(r1, body_com[tid]), r1)
    # body_q_new[tid] = wp.transform(x1, r1)
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
    poses_out: wp.array(dtype=wp.transform),
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
        dq = wp.quat_rotate(q1, I_inv1 * rd) # * 0.5
        w = wp.length(dq)
        # if w > 0.01:
        #     dq = wp.normalize(dq) * 0.01
        delta = -wp.spatial_vector(dq, dp)
        apply_body_delta_positions(body_1, tf1, m_inv1, I_inv1, delta, poses_out)
    if body_2 >= 0 and m_inv2 > 0.0:
        # Eq. 7
        dp = p

        # Eq. 9
        rd = wp.quat_rotate_inv(q2, wp.cross(r2, p))
        dq = wp.quat_rotate(q2, I_inv2 * rd) # * 0.5
        w = wp.length(dq)
        # if w > 0.01:
        #     dq = wp.normalize(dq) * 0.01
        delta = wp.spatial_vector(dq, dp)
        apply_body_delta_positions(body_2, tf2, m_inv2, I_inv2, delta, poses_out)
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
    poses_out: wp.array(dtype=wp.transform),
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
        # if w > 0.01:
        #     dq = wp.normalize(dq) * 0.01
        delta = -wp.spatial_vector(dq, dp)
        apply_body_delta_positions(body_1, tf1, m_inv1, I_inv1, delta, poses_out)
    if body_2 >= 0 and m_inv2 > 0.0:
        rd = n2 * d_lambda
        dq = wp.quat_rotate(q2, I_inv2 * rd) * relaxation
        w = wp.length(dq)
        # if w > 0.01:
        #     dq = wp.normalize(dq) * 0.01
        delta = wp.spatial_vector(dq, dp)
        apply_body_delta_positions(body_2, tf2, m_inv2, I_inv2, delta, poses_out)
    return p

@wp.func
def apply_body_delta_positions(
    tid: int,
    tf: wp.transform,
    inv_m: float,
    inv_I: wp.mat33,
    delta: wp.spatial_vector,
    poses_out: wp.array(dtype=wp.transform),
) -> wp.vec3:
    if inv_m == 0.0:
        return wp.vec3(0.0)

    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)

    # update position
    dp = wp.spatial_bottom(delta)
    p += dp * inv_m

    # update orientation
    dq = wp.spatial_top(delta)
    q += 0.5 * wp.quat(dq, 0.0) * q

    q = wp.normalize(q)

    poses_out[tid] = wp.transform(p, q)
    return wp.vec3(0.0)


@wp.kernel
def apply_body_delta_velocities(
    qd_in: wp.array(dtype=wp.spatial_vector),
    deltas: wp.array(dtype=wp.spatial_vector),
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    qd_out[tid] = qd_in[tid] + deltas[tid]


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
    pose_p = X_pj # wp.transform_identity()
    com_p = wp.vec3(0.0)
    # parent transform and moment arm
    if (id_p >= 0):
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)
    
    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
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
        # a_c = wp.quat_rotate(q_c, axis)
        t_total += act * a_p
    elif (type == wp.sim.JOINT_PRISMATIC):
        a_p = wp.transform_vector(X_wp, axis)
        f_total += act * a_p
    # else:
        # print("joint type not handled in apply_joint_torques")        
        
    # write forces
    if (id_p >= 0):
        wp.atomic_add(body_f, id_p, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total)) 
    wp.atomic_sub(body_f, id_c, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))

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
                      angular_relaxation: float,
                      positional_relaxation: float,
                      dt: float,
                      deltas: wp.array(dtype=wp.spatial_vector)):
    tid = wp.tid()
    type = joint_type[tid]

    if (type == wp.sim.JOINT_FREE):
        return

    # print("--- Joint ---")
    # print(tid)
    
    # rigid body indices of the child and parent
    # id_c = tid
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]
    
    m_inv_p = 0.0
    I_inv_p = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    com_p = wp.vec3(0.0)
    # parent transform and moment arm
    if (id_p >= 0):
        com_p = body_com[id_p]                
        m_inv_p = body_inv_m[id_p]
        I_inv_p = body_inv_I[id_p]
        
    # child transform and moment arm
    com_c = body_com[id_c]
    m_inv_c = body_inv_m[id_c]
    I_inv_c = body_inv_I[id_c]

    # joint properties (for 1D joints)
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    axis = joint_axis[tid]

    if (type == wp.sim.JOINT_FIXED):
        limit_lower = 0.0
        limit_upper = 0.0
    else:
        limit_lower = joint_limit_lower[qd_start]
        limit_upper = joint_limit_upper[qd_start]

    linear_alpha = joint_linear_compliance[tid]
    angular_alpha = joint_angular_compliance[tid]

    linear_alpha_tilde = linear_alpha / dt / dt
    angular_alpha_tilde = angular_alpha / dt / dt

    # prevent division by zero
    linear_alpha_tilde = wp.max(linear_alpha_tilde, 1e-6)
    angular_alpha_tilde = wp.max(angular_alpha_tilde, 1e-6)

    # handle angular constraints
    if (type == wp.sim.JOINT_REVOLUTE):
        X_wp = X_pj
        pose_p = X_pj # wp.transform_identity()
        # parent transform and moment arm
        if (id_p >= 0):
            pose_p = body_q[id_p]
            X_wp = pose_p * X_wp
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)
        
        # child transform and moment arm
        pose_c = body_q[id_c]
        X_wc = pose_c * X_cj
        r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)    

        # local joint rotations
        q_p = wp.transform_get_rotation(X_wp)
        q_c = wp.transform_get_rotation(X_wc)


        # align joint axes
        a_p = wp.quat_rotate(q_p, axis)
        a_c = wp.quat_rotate(q_c, axis)
        # Eq. 20
        corr = wp.cross(a_p, a_c)
        
        angular_correction(
            corr, pose_p, pose_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
            angular_alpha_tilde, angular_relaxation, body_q, id_p, id_c)

        # limit joint angles (Alg. 3)
        pi = 3.14159265359
        two_pi = 2.0 * pi
        if limit_lower > -two_pi or limit_upper < two_pi:
            # find a perpendicular vector to joint axis
            a = axis
            # https://math.stackexchange.com/a/3582461
            g = wp.sign(a[2])
            h = a[2] + g
            b = wp.vec3(g - a[0]*a[0]/h, -a[0]*a[1]/h, -a[0])
            c = wp.normalize(wp.cross(a, b))
            # b = c  # TODO verify

            X_wp = X_pj
            pose_p = X_pj # wp.transform_identity()
            # parent transform and moment arm
            if (id_p >= 0):
                pose_p = body_q[id_p]
                X_wp = pose_p * X_wp
            r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)
            
            # child transform and moment arm
            pose_c = body_q[id_c]
            X_wc = pose_c * X_cj
            r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)    

            # local joint rotations
            q_p = wp.transform_get_rotation(X_wp)
            q_c = wp.transform_get_rotation(X_wc)
            

            # joint axis
            n = wp.quat_rotate(q_p, a)
            # the axes n1 and n2 are aligned with the two bodies
            n1 = wp.quat_rotate(q_p, b)
            n2 = wp.quat_rotate(q_c, b)

            phi = wp.asin(wp.dot(wp.cross(n1, n2), n))
            # print("phi")
            # print(phi)
            if wp.dot(n1, n2) < 0.0:
                phi = pi - phi
            if phi > pi:
                phi -= two_pi
            if phi < -pi:
                phi += two_pi
            if phi < limit_lower or phi > limit_upper:
                phi = wp.clamp(phi, limit_lower, limit_upper)
                # print("clamped phi")
                # print(phi)
                # rot = wp.quat(phi, n[0], n[1], n[2])
                # rot = wp.quat(n, phi)
                rot = wp.quat_from_axis_angle(n, phi)
                n1 = wp.quat_rotate(rot, n1)
                corr = wp.cross(n1, n2) * 0.05
                # print("corr")
                # print(corr)
                # TODO expose
                # angular_alpha_tilde = 0.0001 / dt / dt
                # angular_relaxation = 0.5
                # TODO fix this constraint
                angular_correction(
                    corr, pose_p, pose_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
                    angular_alpha_tilde, angular_relaxation, body_q, id_p, id_c)

        # handle joint targets
        target_ke = joint_target_ke[qd_start]
        if target_ke > 0.0:
            target_angle = joint_target[qd_start]
            # find a perpendicular vector to joint axis
            a = axis
            # https://math.stackexchange.com/a/3582461
            g = wp.sign(a[2])
            h = a[2] + g
            b = wp.vec3(g - a[0]*a[0]/h, -a[0]*a[1]/h, -a[0])
            c = wp.normalize(wp.cross(a, b))
            b = c

            X_wp = X_pj
            pose_p = X_pj # wp.transform_identity()
            # parent transform and moment arm
            if (id_p >= 0):
                pose_p = body_q[id_p]
                X_wp = pose_p * X_wp
            r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)
            
            # child transform and moment arm
            pose_c = body_q[id_c]
            X_wc = pose_c * X_cj
            r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)    

            # local joint rotations
            q_p = wp.transform_get_rotation(X_wp)
            q_c = wp.transform_get_rotation(X_wc)

            
            q = wp.quat_from_axis_angle(a_p, target_angle)
            b_target = wp.quat_rotate(q, wp.quat_rotate(q_p, b))
            b2 = wp.quat_rotate(q_c, b)
            # Eq. 21
            d_target = wp.cross(b_target, b2)

            target_compliance = 1.0 / target_ke / dt / dt
            angular_correction(
                d_target, pose_p, pose_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
                target_compliance, angular_relaxation, body_q, id_p, id_c)

    if (type == wp.sim.JOINT_FIXED) or (type == wp.sim.JOINT_PRISMATIC):
        # align the mutual orientations of the two bodies
        X_wp = X_pj
        pose_p = X_pj # wp.transform_identity()
        # parent transform and moment arm
        if (id_p >= 0):
            pose_p = body_q[id_p]
            X_wp = pose_p * X_wp
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)
        
        # child transform and moment arm
        pose_c = body_q[id_c]
        X_wc = pose_c * X_cj
        r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)    

        # local joint rotations
        q_p = wp.transform_get_rotation(X_wp)
        q_c = wp.transform_get_rotation(X_wc)

        # Eq. 18-19
        q = q_p * wp.quat_inverse(q_c)
        corr = 2.0 * wp.vec3(q[0], q[1], q[2])
        angular_correction(
            -corr, pose_p, pose_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
            angular_alpha_tilde, angular_relaxation, body_q, id_p, id_c)
    # if (type == wp.sim.JOINT_BALL):
    #     # TODO: implement Eq. 22-24
    #     wp.print("Ball joint not implemented")
    #     return

    # handle positional constraints
    X_wp = X_pj
    pose_p = X_pj # wp.transform_identity()
    # parent transform and moment arm
    if (id_p >= 0):
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)
    
    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)    

    # local joint rotations
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # joint connection points
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    
    # compute error between the joint attachment points on both bodies
    # delta x is the difference of point r_2 minus point r_1 (Fig. 3)
    dx = x_c - x_p

    # define set of perpendicular unit axes a, b, c
    # (Sec. 3.4.1)
    normal_a = wp.vec3(1.0, 0.0, 0.0)
    normal_b = wp.vec3(0.0, 1.0, 0.0)
    normal_c = wp.vec3(0.0, 0.0, 1.0)

    # rotate the error vector into the joint frame
    q_dx = q_p
    # q_dx = q_c
    # q_dx = wp.transform_get_rotation(pose_p)
    dx = wp.quat_rotate_inv(q_dx, dx)

    lower_pos_limits = wp.vec3(0.0)
    upper_pos_limits = wp.vec3(0.0)
    if (type == wp.sim.JOINT_PRISMATIC):
        lower_pos_limits = axis * limit_lower
        upper_pos_limits = axis * limit_upper

    corr = wp.vec3(0.0)

    d = wp.dot(normal_a, dx)
    if (d < lower_pos_limits[0]):
        corr -= normal_a * (lower_pos_limits[0] - d)
    if (d > upper_pos_limits[0]):
        corr -= normal_a * (upper_pos_limits[0] - d)

    d = wp.dot(normal_b, dx)
    if (d < lower_pos_limits[1]):
        corr -= normal_b * (lower_pos_limits[1] - d)
    if (d > upper_pos_limits[1]):
        corr -= normal_b * (upper_pos_limits[1] - d)

    d = wp.dot(normal_c, dx)
    if (d < lower_pos_limits[2]):
        corr -= normal_c * (lower_pos_limits[2] - d)
    if (d > upper_pos_limits[2]):
        corr -= normal_c * (upper_pos_limits[2] - d)

    # rotate correction vector into world frame
    corr = wp.quat_rotate(q_dx, corr)

    # print("parent")
    # print(id_p)
    # print("dx")
    # print(dx)
    # print("r_p")
    # print(r_p)
    # print("r_c")
    # print(r_c)
    
    # handle positional constraints (Sec. 3.3.1)
    positional_correction(corr, r_p, r_c, pose_p, pose_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
                          linear_alpha_tilde, positional_relaxation, 0.0, 0.0, body_q, id_p, id_c)

    # if (tid == 0):
    #     print("x_err")
    #     print(dx)
    #     print("corr")
    #     print(corr)
    #     # print("delta 1")
    #     # print(deltas[id_p])
    #     print("delta 2")
    #     print(deltas[id_c])


@wp.kernel
def solve_body_contact_positions(
    body_q_new: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),
    contact_dist: wp.array(dtype=float),
    contact_mat: wp.array(dtype=int),
    materials: wp.array(dtype=wp.vec4),
    normal_relaxation: float,
    friction_relaxation: float,
    contact_n_lambda: wp.array(dtype=float),
    contact_t_lambda: wp.array(dtype=float)
):

    tid = wp.tid()

    c_body = contact_body[tid]
    c_point = contact_point[tid]
    c_dist = contact_dist[tid]
    c_mat = contact_mat[tid]

    X_new = body_q_new[c_body]
    X_prev = body_q_prev[c_body]

    n = wp.vec3(0.0, 1.0, 0.0)

    # transform point to world space
    p_new = wp.transform_point(X_new, c_point) - n * c_dist # add on 'thickness' of shape, e.g.: radius of sphere/capsule
    p_prev = wp.transform_point(X_prev, c_point) - n * c_dist # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    # moment arm around center of mass
    r_new = p_new - wp.transform_point(X_new, body_com[c_body])
    r_prev = p_prev - wp.transform_point(X_prev, body_com[c_body])

    # ground penetration depth
    d = wp.dot(n, p_new)

    
    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    mat = materials[c_mat]

    ke = mat[0]       # restitution coefficient
    kd = mat[1]       # damping coefficient
    kf = mat[2]       # friction coefficient
    mu = mat[3]       # coulomb friction

    if d < 0.0:
        # print("contact!")
        dx = -d * n
        m_inv = body_m_inv[c_body]
        I_inv = body_I_inv[c_body]
        # correct_body(dx, r_new, X_new, I_inv, False, deltas, c_body)
        null_tf = wp.transform_identity()
        null_inv_mass = 0.0
        null_inv_I = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        lambda_n = contact_n_lambda[tid]
        p_n = positional_correction(dx, r_new, r_new, X_new, null_tf, m_inv,
                              null_inv_mass, I_inv, null_inv_I,
                              0.0, normal_relaxation, lambda_n, 0.0, body_q_new, c_body, -1)
        lambda_n += wp.length(p_n)
        contact_n_lambda[tid] = lambda_n

        # use update poses        
        X_new = body_q_new[c_body]
        # transform point to world space
        p_new = wp.transform_point(X_new, c_point) - n * c_dist # add on 'thickness' of shape, e.g.: radius of sphere/capsule

        # moment arm around center of mass
        r_new = p_new - wp.transform_point(X_new, body_com[c_body])

        # ground penetration depth
        d = wp.dot(n, p_new)


        # handle static friction (Eq. 27-28)
        dp = p_prev - p_new
        dpt = dp - wp.dot(dp, n) * n
        max_lambda = lambda_n * mu
        # print("max_lambda")
        # print(max_lambda)
        lambda_t = contact_t_lambda[tid]
        p_t = positional_correction(dpt, r_new, r_new, X_new, null_tf, m_inv,
                              null_inv_mass, I_inv, null_inv_I,
                              0.0, friction_relaxation, lambda_t, max_lambda, body_q_new, c_body, -1)

        # Eq. 14
        lambda_t += wp.length(p_t)
        contact_t_lambda[tid] = lambda_t
    # else:
    #     contact_lambda[tid] = 0.0


@wp.kernel
def solve_body_contact_velocities(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),
    contact_dist: wp.array(dtype=float),
    contact_mat: wp.array(dtype=int),
    contact_n_lambda: wp.array(dtype=float),
    materials: wp.array(dtype=wp.vec4),
    dt: float,
    deltas: wp.array(dtype=wp.spatial_vector)
):

    tid = wp.tid()

    c_body = contact_body[tid]
    c_point = contact_point[tid]
    c_dist = contact_dist[tid]
    c_mat = contact_mat[tid]

    tf = body_q[c_body]

    n = wp.vec3(0.0, 1.0, 0.0)

    # transform point to world space
    p = wp.transform_point(tf, c_point) - n * c_dist # add on 'thickness' of shape, e.g.: radius of sphere/capsule
   
    # moment arm around center of mass
    r = p - wp.transform_point(tf, body_com[c_body])

    # ground penetration depth
    # d = wp.dot(n, p)
    # if d > 0.0:
    #     return
    
    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    mat = materials[c_mat]

    ke = mat[0]       # restitution coefficient
    kd = mat[1]       # damping coefficient
    kf = mat[2]       # friction coefficient
    mu = mat[3]       # coulomb friction

    # if d < 0.0:
    v0 = wp.spatial_bottom(body_qd[c_body])
    w0 = wp.spatial_top(body_qd[c_body])

    # contact point velocity
    v = v0 + wp.cross(w0, r)        # this is body velocity cross offset, so it's the velocity of the contact point.
    # Eq. 29
    vn = wp.dot(n, v)
    vt = v - n * vn

    lambda_n = contact_n_lambda[tid]
    if lambda_n == 0.0:
        return

    # contact normal force
    fn = lambda_n / dt / dt

    # velocity update to handle friction (Eq. 30)
    lvt = wp.length(vt)
    if lvt == 0.0:
        return

    dv = -vt / lvt * wp.min(dt * mu * fn, lvt)
    
    # print("dv")
    # print(dv)

    # apply velocity correction (Eq. 33)
    n = wp.normalize(dv)

    q1 = wp.transform_get_rotation(tf)

    # Eq. 2-3 (make sure to project into the frame of the body)
    r1xn = wp.quat_rotate_inv(q1, wp.cross(r, n))

    m_inv = body_m_inv[c_body]
    I_inv = body_I_inv[c_body]
    w1 = m_inv + wp.dot(r1xn, I_inv * r1xn)
    w2 = 0.0
    w = w1 + w2

    if w == 0.0:
        return

    p = dv / w

    # print("p")
    # print(p)

    lin = p * m_inv
    ang = I_inv * wp.quat_rotate_inv(q1, wp.cross(r, p))
    ang = wp.quat_rotate(q1, ang) * 0.25

    wp.atomic_add(deltas, c_body, wp.spatial_vector(ang, lin))
    # wp.atomic_add(deltas, c_body, wp.spatial_vector(ang, lin) * 0.4)
    # wp.atomic_add(body_qd_out, c_body, wp.spatial_vector(ang, lin) * 0.1)
    # wp.atomic_add(body_qd_out, c_body, wp.spatial_vector(ang, lin))
    # print("update qd")
    # print(wp.spatial_vector(ang, lin))

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
        # print("dq[3] < 0.0 in update_body_velocities")
        # print("omega:")
        # print(omega)
        # print("linear v:")
        # print(v)
        # print("q:")
        # print(q)
        # print("q_prev:")
        # print(q_prev)
        omega = -omega

    qd_out[tid] = wp.spatial_vector(omega, v)

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
                 joint_angular_relaxation=1.0,
                 contact_normal_relaxation=1.0,
                 contact_friction_relaxation=1.0):

        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation

        self.joint_positional_relaxation = joint_positional_relaxation
        self.joint_angular_relaxation = joint_angular_relaxation

        self.contact_normal_relaxation = contact_normal_relaxation
        self.contact_friction_relaxation = contact_friction_relaxation

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
                        dt,
                    ],
                    outputs=[
                        state_out.body_q,
                        state_out.body_qd
                    ],
                    device=model.device)


                # -------------------------------------
                # integrate bodies
                if (model.contact_count):
                    state_out.contact_n_lambda.zero_()
                    state_out.contact_t_lambda.zero_()

                for i in range(self.iterations):
                    # print(f"### iteration {i} / {self.iterations-1}")
                    state_out.body_deltas.zero_()

                    wp.launch(kernel=solve_body_joints,
                              dim=model.joint_count,
                              inputs=[
                                  state_out.body_q,
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
                                  self.joint_angular_relaxation,
                                  self.joint_positional_relaxation,
                                  dt
                              ],
                              outputs=[
                                  state_out.body_deltas
                              ],
                              device=model.device)

                    
                    if (model.contact_count):
                        wp.launch(kernel=solve_body_contact_positions,
                                dim=model.contact_count,
                                inputs=[
                                    state_out.body_q,
                                    state_out.body_q_prev,
                                    model.body_com,
                                    model.body_inv_mass,
                                    model.body_inv_inertia,
                                    model.contact_body0,
                                    model.contact_point0,
                                    model.contact_dist,
                                    model.contact_material,
                                    model.shape_materials,
                                    self.contact_normal_relaxation,
                                    self.contact_friction_relaxation,
                                ],
                                outputs=[
                                    # state_out.body_deltas,
                                    state_out.contact_n_lambda,
                                    state_out.contact_t_lambda
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

                if (model.contact_count > 0 and model.ground):
                    state_out.body_deltas.zero_()
                    wp.launch(kernel=solve_body_contact_velocities,
                            dim=model.contact_count,
                            inputs=[
                                state_out.body_q,
                                # state_out.body_q_prev,
                                state_out.body_qd,
                                model.body_com,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.contact_body0,
                                model.contact_point0,
                                model.contact_dist,
                                model.contact_material,
                                state_out.contact_n_lambda,
                                model.shape_materials,
                                dt 
                            ],
                            outputs=[
                                state_out.body_deltas
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

            # state_out.body_q = body_q_pred
            # state_out.body_qd = body_qd_pred

            return state_out
