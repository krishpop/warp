# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""A module for building simulation models and state.
"""

import warp as wp

@wp.func
def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if (d1 <= 0.0 and d2 <= 0.0):
        return wp.vec3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    if (d3 >= 0.0 and d4 <= d3):
        return wp.vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
        return wp.vec3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)

    if (d6 >= 0.0 and d5 <= d6):
        return wp.vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
        return wp.vec3(0.0, w, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return wp.vec3(1.0 - v - w, v, w)

@wp.func
def sphere_sdf(center: wp.vec3, radius: float, p: wp.vec3):

    return wp.length(p-center) - radius

@wp.func
def sphere_sdf_grad(center: wp.vec3, radius: float, p: wp.vec3):

    return wp.normalize(p-center)

@wp.func
def box_sdf(upper: wp.vec3, p: wp.vec3):

    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    qx = abs(p[0])-upper[0]
    qy = abs(p[1])-upper[1]
    qz = abs(p[2])-upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))
    
    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def box_sdf_grad(upper: wp.vec3, p: wp.vec3):

    qx = abs(p[0])-upper[0]
    qy = abs(p[1])-upper[1]
    qz = abs(p[2])-upper[2]

    # exterior case
    if (qx > 0.0 or qy > 0.0 or qz > 0.0):
        
        x = wp.clamp(p[0], 0.0-upper[0], upper[0])
        y = wp.clamp(p[1], 0.0-upper[1], upper[1])
        z = wp.clamp(p[2], 0.0-upper[2], upper[2])

        return wp.normalize(p - wp.vec3(x, y, z))

    sx = wp.sign(p[0])
    sy = wp.sign(p[1])
    sz = wp.sign(p[2])

    # x projection
    if (qx > qy and qx > qz):
        return wp.vec3(sx, 0.0, 0.0)
    
    # y projection
    if (qy > qx and qy > qz):
        return wp.vec3(0.0, sy, 0.0)

    # z projection
    if (qz > qx and qz > qy):
        return wp.vec3(0.0, 0.0, sz)

@wp.func
def capsule_sdf(radius: float, half_width: float, p: wp.vec3):

    if (p[0] > half_width):
        return wp.length(wp.vec3(p[0] - half_width, p[1], p[2])) - radius

    if (p[0] < 0.0 - half_width):
        return wp.length(wp.vec3(p[0] + half_width, p[1], p[2])) - radius

    return wp.length(wp.vec3(0.0, p[1], p[2])) - radius

@wp.func
def capsule_sdf_grad(radius: float, half_width: float, p: wp.vec3):

    if (p[0] > half_width):
        return wp.normalize(wp.vec3(p[0] - half_width, p[1], p[2]))

    if (p[0] < 0.0 - half_width):
        return wp.normalize(wp.vec3(p[0] + half_width, p[1], p[2]))
        
    return wp.normalize(wp.vec3(0.0, p[1], p[2]))

@wp.func
def plane_sdf(width: float, length: float, p: wp.vec3):
    # SDF for a quad in the xz plane
    d = wp.max(wp.abs(p[0]) - width, wp.abs(p[2]) - length)
    d = wp.max(d, abs(p[1]))
    return d

@wp.func
def closest_point_line_segment(a: wp.vec3, b: wp.vec3, point: wp.vec3) -> wp.vec3:
    ab = b - a
    ap = point - a
    t = wp.dot(ap, ab) / wp.dot(ab, ab)
    t = wp.clamp(t, 0.0, 1.0)
    return a + t * ab

@wp.kernel
def create_soft_contacts(
    num_particles: int,
    particle_x: wp.array(dtype=wp.vec3), 
    body_X_wb: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_geo_type: wp.array(dtype=int), 
    shape_geo_id: wp.array(dtype=wp.uint64),
    shape_geo_scale: wp.array(dtype=wp.vec3),
    margin: float,
    #outputs,
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_body: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    soft_contact_max: int):
    
    tid = wp.tid()           

    shape_index = tid // num_particles     # which shape
    particle_index = tid % num_particles   # which particle
    rigid_index = shape_body[shape_index]

    px = particle_x[particle_index]

    X_wb = wp.transform_identity()
    if (rigid_index >= 0):
        X_wb = body_X_wb[rigid_index]
    
    X_bs = shape_X_bs[shape_index]

    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)
    
    # transform particle position to shape local space
    x_local = wp.transform_point(X_sw, px)

    # geo description
    geo_type = shape_geo_type[shape_index]
    geo_scale = shape_geo_scale[shape_index]

    # evaluate shape sdf
    d = 1.e+6 
    n = wp.vec3()
    v = wp.vec3()

    # GEO_SPHERE (0)
    if (geo_type == 0):
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
        n = sphere_sdf_grad(wp.vec3(), geo_scale[0], x_local)

    # GEO_BOX (1)
    if (geo_type == 1):
        d = box_sdf(geo_scale, x_local)
        n = box_sdf_grad(geo_scale, x_local)
        
    # GEO_CAPSULE (2)
    if (geo_type == 2):
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
        n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    # GEO_MESH (3)
    if (geo_type == 3):
        mesh = shape_geo_id[shape_index]

        face_index = int(0)
        face_u = float(0.0)  
        face_v = float(0.0)
        sign = float(0.0)

        if (wp.mesh_query_point(mesh, x_local/geo_scale[0], margin, sign, face_index, face_u, face_v)):

            shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
            shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)

            shape_p = shape_p*geo_scale[0]
            shape_v = shape_v*geo_scale[0]

            delta = x_local-shape_p
            d = wp.length(delta)*sign
            n = wp.normalize(delta)*sign
            v = shape_v


    if (d < margin):

        index = wp.atomic_add(soft_contact_count, 0, 1) 

        if (index < soft_contact_max):

            # compute contact point in body local space
            body_pos = wp.transform_point(X_bs, x_local - n*d)
            body_vel = wp.transform_vector(X_bs, v)

            world_normal = wp.transform_vector(X_ws, n)

            soft_contact_body[index] = rigid_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = body_vel
            soft_contact_particle[index] = particle_index
            soft_contact_normal[index] = world_normal

    # GEO_PLANE (5)
    if (geo_type == 5):
        d = plane_sdf(geo_scale[0], geo_scale[1], x_local)
        n = wp.vec3(0.0, 1.0, 0.0)


@wp.func
def volume_grad(volume: wp.uint64,
                p: wp.vec3):
    
    eps = 0.05  # TODO make this a parameter
    q = wp.volume_world_to_index(volume, p)

    # compute gradient of the SDF using finite differences
    dx = wp.volume_sample_f(volume, q + wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(volume, q - wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR)
    dy = wp.volume_sample_f(volume, q + wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(volume, q - wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR)
    dz = wp.volume_sample_f(volume, q + wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR) - wp.volume_sample_f(volume, q - wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR)

    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.kernel
def update_rigid_ground_contacts(
    ground_plane: wp.array(dtype=float),
    rigid_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    contact_point_ref: wp.array(dtype=wp.vec3),
    ground_contact_shape: wp.array(dtype=int),
    shape_contact_thickness: wp.array(dtype=float),
    rigid_contact_margin: float,
    rigid_contact_max: int,
    # outputs
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
    contact_thickness: wp.array(dtype=float)
):
    tid = wp.tid()
    body = rigid_body[tid]
    shape = ground_contact_shape[tid]
    thickness = shape_contact_thickness[shape]
    X_wb = body_q[body]
    X_bw = wp.transform_inverse(X_wb)
    X_bs = shape_X_bs[shape]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    n = wp.vec3(ground_plane[0], ground_plane[1], ground_plane[2])
    p_ref = wp.transform_point(X_ws, contact_point_ref[tid])
    c = ground_plane[3]  # ground plane offset
    d = wp.dot(p_ref, n) - c
    if (d < thickness + rigid_contact_margin):
        index = wp.inc_index(contact_count, tid, rigid_contact_max)
        if (index >= 0):
            contact_point0[index] = wp.transform_point(X_bw, p_ref)
            # project contact point onto ground plane
            contact_point1[index] = p_ref - n*d
            contact_body0[index] = body
            contact_body1[index] = -1
            contact_offset0[index] = wp.transform_vector(X_bw, -thickness * n)
            contact_offset1[index] = wp.vec3(0.0)
            contact_normal[index] = n
            contact_shape0[index] = shape
            contact_shape1[index] = -1
            contact_thickness[index] = thickness
        else:
            print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")

@wp.kernel
def broadphase_collision_pairs(
    contact_pairs: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_geo_type: wp.array(dtype=int),
    collision_radius: wp.array(dtype=float),
    rigid_contact_max: int,
    mesh_num_points: wp.array(dtype=int),
    rigid_contact_margin: float,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),    
    contact_shape1: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
):
    tid = wp.tid()
    shape_a = contact_pairs[tid, 0]
    shape_b = contact_pairs[tid, 1]

    rigid_a = shape_body[shape_a]
    rigid_b = shape_body[shape_b]

    X_wb_a = body_q[rigid_a]
    X_wb_b = body_q[rigid_b]
    
    X_bs_a = shape_X_bs[shape_a]
    X_bs_b = shape_X_bs[shape_b]

    X_ws_a = wp.transform_multiply(X_wb_a, X_bs_a)
    X_ws_b = wp.transform_multiply(X_wb_b, X_bs_b)

    p_a = wp.transform_get_translation(X_ws_a)
    p_b = wp.transform_get_translation(X_ws_b)

    d = wp.length(p_a - p_b)
    r_a = collision_radius[shape_a]
    r_b = collision_radius[shape_b]
    # if d > r_a + r_b + rigid_contact_margin:
    #     # print("too far")
    #     return

    type_a = shape_geo_type[shape_a]
    type_b = shape_geo_type[shape_b]
    # unique ordering of shape pairs
    if type_a < type_b:
        actual_shape_a = shape_a
        actual_shape_b = shape_b
        actual_type_a = type_a
    else:
        actual_shape_a = shape_b
        actual_shape_b = shape_a
        actual_type_a = type_b

    # determine how many contact points need to be evaluated
    num_contacts = 0
    if actual_type_a == wp.sim.GEO_SPHERE:
        num_contacts = 1
    elif actual_type_a == wp.sim.GEO_CAPSULE:
        num_contacts = 2
    elif actual_type_a == wp.sim.GEO_MESH:
        num_contacts_a = mesh_num_points[shape_a]
        num_contacts_b = mesh_num_points[shape_b]
        num_contacts = num_contacts_a + num_contacts_b
        if num_contacts > 0:
            index = wp.atomic_add(contact_count, 0, num_contacts)
            # allocate contact points from mesh A against B
            for i in range(num_contacts_a):
                if index + i >= rigid_contact_max:
                    print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                    return
                contact_shape0[index + i] = shape_a
                contact_shape1[index + i] = shape_b
                contact_point_id[index + i] = i
            # allocate contact points from mesh B against A
            for i in range(num_contacts_b):
                if index + num_contacts_a + i >= rigid_contact_max:
                    print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                    return
                contact_shape0[index + num_contacts_a + i] = shape_b
                contact_shape1[index + num_contacts_a + i] = shape_a
                contact_point_id[index + num_contacts_a + i] = i
        return
    else:
        print("broadphase_collision_pairs: unsupported geometry type")

    if num_contacts > 0:
        index = wp.atomic_add(contact_count, 0, num_contacts)
        # allocate contact points
        for i in range(num_contacts):
            if index + i >= rigid_contact_max:
                print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                return
            contact_shape0[index + i] = actual_shape_a
            contact_shape1[index + i] = actual_shape_b
            contact_point_id[index + i] = i


@wp.kernel
def handle_contact_pairs(
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_geo_type: wp.array(dtype=int), 
    shape_geo_id: wp.array(dtype=wp.uint64),
    shape_geo_scale: wp.array(dtype=wp.vec3),
    shape_contact_thickness: wp.array(dtype=float),
    rigid_contact_margin: float,
    contact_shape0: wp.array(dtype=int),    
    contact_shape1: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
    rigid_contact_count: wp.array(dtype=int),
    # outputs
    contact_body0: wp.array(dtype=int),
    contact_body1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_thickness: wp.array(dtype=float)):

    tid = wp.tid()
    if tid >= rigid_contact_count[0]:
        return
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return

    point_id = contact_point_id[tid]

    rigid_a = shape_body[shape_a]
    rigid_b = shape_body[shape_b]
    
    # fill in contact rigid body ids
    contact_body0[tid] = rigid_a
    contact_body1[tid] = rigid_b

    X_wb_a = body_q[rigid_a]
    X_wb_b = body_q[rigid_b]
    
    X_bs_a = shape_X_bs[shape_a]
    X_bs_b = shape_X_bs[shape_b]

    X_ws_a = wp.transform_multiply(X_wb_a, X_bs_a)
    X_ws_b = wp.transform_multiply(X_wb_b, X_bs_b)
    
    X_sw_a = wp.transform_inverse(X_ws_a)
    X_sw_b = wp.transform_inverse(X_ws_b)

    X_bw_a = wp.transform_inverse(X_wb_a)
    X_bw_b = wp.transform_inverse(X_wb_b)

    # geo description
    geo_type_a = shape_geo_type[shape_a]
    geo_scale_a = shape_geo_scale[shape_a]
    geo_type_b = shape_geo_type[shape_b]
    geo_scale_b = shape_geo_scale[shape_b]

    if geo_type_a == wp.sim.GEO_SPHERE:
        p_a = wp.transform_get_translation(X_ws_a)
        if geo_type_b == wp.sim.GEO_SPHERE:
            radius_a = geo_scale_a[0]
            radius_b = geo_scale_b[0]
            p_b = wp.transform_get_translation(X_ws_b)
        elif geo_type_b == wp.sim.GEO_CAPSULE:
            radius_a = geo_scale_a[0]
            radius_b = geo_scale_b[0]
            half_width_b = geo_scale_b[1]
            # capsule B
            A_b = wp.transform_point(X_ws_b, wp.vec3(half_width_b, 0.0, 0.0))
            B_b = wp.transform_point(X_ws_b, wp.vec3(-half_width_b, 0.0, 0.0))
            p_b = closest_point_line_segment(A_b, B_b, p_a)
        elif geo_type_b == wp.sim.GEO_MESH:
            mesh_b = shape_geo_id[shape_b]
            query_b_local = wp.transform_point(X_sw_b, p_a)
            face_index = int(0)
            face_u = float(0.0)  
            face_v = float(0.0)
            sign = float(0.0)
            res = wp.mesh_query_point(mesh_b, query_b_local/geo_scale_b[0], 0.15, sign, face_index, face_u, face_v)
            if (res):
                shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
                shape_p = shape_p*geo_scale_b[0]
                p_b = wp.transform_point(X_ws_b, shape_p)
            else:
                contact_shape0[tid] = -1
                contact_shape1[tid] = -1
                return
        else:
            print("Unsupported geometry type in sphere collision handling")
            print(geo_type_b)
            return
            
        thickness_a = shape_contact_thickness[shape_a]
        thickness_b = shape_contact_thickness[shape_b]
        thickness = thickness_a + thickness_b
        diff = p_a - p_b
        d = wp.length(diff) - thickness
        normal = wp.normalize(diff)
        if (d < rigid_contact_margin):
            # transform from world into body frame (so the contact point includes the shape transform)
            contact_point0[tid] = wp.transform_point(X_bw_a, p_a)  # might not be zero if shape has transform
            contact_point1[tid] = wp.transform_point(X_bw_b, p_b)
            contact_offset0[tid] = wp.transform_vector(wp.transform_inverse(X_bw_a), -thickness_a * normal)
            contact_offset1[tid] = wp.transform_vector(wp.transform_inverse(X_bw_b), thickness_b * normal)
            contact_normal[tid] = normal
            contact_thickness[tid] = thickness
        else:
            contact_shape0[tid] = -1
            contact_shape1[tid] = -1
        return
            
    if (geo_type_a == wp.sim.GEO_CAPSULE and geo_type_b == wp.sim.GEO_CAPSULE):
        # if point_id == 1:
        #     # TODO skip second capsule point for now
        #     contact_shape0[tid] = -1
        #     contact_shape1[tid] = -1
        #     print("Skipping second capsule point")
        #     return
        # https://wickedengine.net/2020/04/26/capsule-collision-detection/
        radius_a = geo_scale_a[0]
        radius_b = geo_scale_b[0]
        # capsule extends along x axis
        half_width_a = geo_scale_a[1]
        half_width_b = geo_scale_b[1]

        # capsule A
        A_a = wp.transform_point(X_ws_a, wp.vec3(half_width_a, 0.0, 0.0))
        B_a = wp.transform_point(X_ws_a, wp.vec3(-half_width_a, 0.0, 0.0))
        
        # capsule B
        A_b = wp.transform_point(X_ws_b, wp.vec3(half_width_b, 0.0, 0.0))
        B_b = wp.transform_point(X_ws_b, wp.vec3(-half_width_b, 0.0, 0.0))

        # squared distances between line endpoints
        d0 = wp.length_sq(A_b - A_a)
        d1 = wp.length_sq(B_b - A_a)
        d2 = wp.length_sq(A_b - B_a)
        d3 = wp.length_sq(B_b - B_a)

        # select best potential endpoint on capsule A
        best_A = A_a
        if ((d3 < d0 and d3 < d1) or (d2 < d1 and d2 < d0)):
            best_A = B_a
        
        # select point on capsule B line segment nearest to best potential endpoint on A capsule:
        best_B = closest_point_line_segment(A_b, B_b, best_A)
        
        if point_id == 1:
            # now do the same for capsule A segment:
            best_A = closest_point_line_segment(A_a, B_a, best_B)

        thickness = shape_contact_thickness[shape_a] + shape_contact_thickness[shape_b]

        diff = best_A - best_B
        # print("best A, B:")
        # print(best_A)
        # print(best_B)
        d = wp.length(diff) - thickness
        if (d < rigid_contact_margin):
            normal = wp.normalize(diff)
            # transform from world into body frame (so the contact point includes the shape transform)
            contact_point0[tid] = wp.transform_point(X_bw_a, best_A)
            contact_point1[tid] = wp.transform_point(X_bw_b, best_B)
            # TODO verify radius is correct
            contact_offset0[tid] = wp.transform_vector(X_bw_a, -radius_a * normal)
            contact_offset1[tid] = wp.transform_vector(X_bw_b, radius_b * normal)
            contact_normal[tid] = normal
            contact_thickness[tid] = thickness
        else:
            contact_shape0[tid] = -1
            contact_shape1[tid] = -1
        return

    if geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_MESH:
        mesh_a = shape_geo_id[shape_a]
        mesh_b = shape_geo_id[shape_b]

        body_a_pos = wp.mesh_get_point(mesh_a, point_id) * geo_scale_a[0]
        p_world = wp.transform_point(X_ws_a, body_a_pos)
        query_b_local = wp.transform_point(X_sw_b, p_world)

        face_index = int(0)
        face_u = float(0.0)  
        face_v = float(0.0)
        sign = float(0.0)

        res = wp.mesh_query_point(mesh_b, query_b_local/geo_scale_b[0], rigid_contact_margin, sign, face_index, face_u, face_v)

        if (res):
            shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
            shape_p = shape_p*geo_scale_b[0]
            # contact direction vector in frame of body B
            diff_b = query_b_local - shape_p
            d = wp.length(diff_b) * sign
            n = wp.normalize(diff_b) * sign
        else:
            contact_shape0[tid] = -1
            contact_shape1[tid] = -1
            return

        thickness_a = shape_contact_thickness[shape_a]
        thickness_b = shape_contact_thickness[shape_b]
        thickness = thickness_a + thickness_b
        err = d - thickness
        if (err < rigid_contact_margin):
            # compute point at the surface of mesh b
            body_b_pos = shape_p - n*err

            # offset by contact thickness to be used in PBD contact friction constraints
            contact_offset0[tid] = wp.transform_vector(wp.transform_inverse(X_bw_a), -thickness_a * n)
            contact_offset1[tid] = wp.transform_vector(wp.transform_inverse(X_bw_b), thickness_b * n)

            # assign contact points in body local spaces
            contact_point0[tid] = body_a_pos            
            contact_point1[tid] = body_b_pos

            # convert n to world frame
            n = wp.transform_vector(X_ws_b, n)
            contact_normal[tid] = n

            contact_thickness[tid] = thickness
        else:
            contact_shape0[tid] = -1
            contact_shape1[tid] = -1
        return
        
    print("Unsupported geometry type in collision handling")
    return

@wp.kernel
def create_mesh_sdf_contacts(
    shape_a: int,
    shape_b: int,
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_geo_type: wp.array(dtype=int), 
    shape_geo_id: wp.array(dtype=wp.uint64),
    shape_volume_id: wp.array(dtype=wp.uint64),
    shape_geo_scale: wp.array(dtype=wp.vec3),
    shape_contact_thickness: wp.array(dtype=float),
    contact_max: int,
    rigid_contact_margin: float,
    #outputs
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
    contact_thickness: wp.array(dtype=float)):
    
    tid = wp.tid()           

    rigid_a = shape_body[shape_a]
    rigid_b = shape_body[shape_b]

    X_wb_a = body_q[rigid_a]
    X_wb_b = body_q[rigid_b]
    
    X_bs_a = shape_X_bs[shape_a]
    X_bs_b = shape_X_bs[shape_b]

    X_ws_a = wp.transform_multiply(X_wb_a, X_bs_a)
    X_sw_a = wp.transform_inverse(X_ws_a)
    X_ws_b = wp.transform_multiply(X_wb_b, X_bs_b)
    X_sw_b = wp.transform_inverse(X_ws_b)

    # geo description
    geo_type_a = shape_geo_type[shape_a]
    geo_scale_a = shape_geo_scale[shape_a]
    geo_type_b = shape_geo_type[shape_b]
    geo_scale_b = shape_geo_scale[shape_b]

    # evaluate shape sdf
    d = 1.e+6 
    n = wp.vec3()
    v = wp.vec3()

    # GEO_SPHERE (0)
    # if (geo_type_a == 0 and geo_type_b == 0):
    #     d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
    #     n = sphere_sdf_grad(wp.vec3(), geo_scale[0], x_local)

    # # GEO_BOX (1)
    # if (geo_type == 1):
    #     d = box_sdf(geo_scale, x_local)
    #     n = box_sdf_grad(geo_scale, x_local)
        
    # # GEO_CAPSULE (2)
    # if (geo_type == 2):
    #     d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
    #     n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    # GEO_MESH (3)
    if (geo_type_a == 3 and geo_type_b == 3):
        # mesh vertex <> volume contact
        mesh_a = shape_geo_id[shape_a]
        mesh_b = shape_geo_id[shape_b]
        
        # print(volume)

        # print(mesh)
        # print("mesh")
        # print(mesh)
        # print("point")
        # print(tid)

        body_a_pos = wp.mesh_get_point(mesh_a, tid) * geo_scale_a[0]
        p_world = wp.transform_point(X_ws_a, body_a_pos)

        # print("body_a_pos")
        # print(body_a_pos)
        # print(tid)

        query_b_local = wp.transform_point(X_sw_b, p_world)

        # print("p_mesh")
        # print(p_mesh)
        # print("p_world")
        # print(p_world)
        # print("p_vol")
        # print(p_vol)

        d = float(0.0)

        # toggle between volume- or mesh-based collision
        if False:
            # transform point to world space   
            volume = shape_volume_id[shape_b]
            p_vol_local = wp.volume_world_to_index(volume, query_b_local) #* 0.2
            
            # print("query volume")
            # print(volume)
            d = wp.volume_sample_f(volume, p_vol_local, wp.Volume.LINEAR)
            # print("query volume grad")
            n = volume_grad(volume, query_b_local)
            shape_p = query_b_local

            # print("d")
            # print(d)

        else:

            face_index = int(0)
            face_u = float(0.0)  
            face_v = float(0.0)
            sign = float(0.0)

            res = wp.mesh_query_point(mesh_b, query_b_local/geo_scale_b[0], 0.15, sign, face_index, face_u, face_v)

            if (res):
                shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
                shape_p = shape_p*geo_scale_b[0]

                delta = query_b_local-shape_p
                d = wp.length(delta)*sign
                n = wp.normalize(delta)*sign

        # rigid_contact_thickness = 0.0 #0.0001
        thickness_a = shape_contact_thickness[shape_a]
        thickness_b = shape_contact_thickness[shape_b]
        thickness = thickness_a + thickness_b

        if (d < thickness + rigid_contact_margin):
            # increment contact count
            index = wp.inc_index(contact_count, tid, contact_max)

            if (index >= 0):
                err = d - thickness

                # mesh collision
                # compute point at the surface of volume b
                body_b_pos = shape_p - n*err

                # offset by contact thickness to be used in PBD contact friction constraints
                contact_offset0[index] = wp.transform_vector(wp.transform_inverse(X_sw_a), -thickness_a * n)
                contact_offset1[index] = wp.transform_vector(wp.transform_inverse(X_sw_b), thickness_b * n)

                # assign contact points in body local spaces
                contact_body0[index] = rigid_a
                contact_point0[index] = body_a_pos
                
                contact_body1[index] = rigid_b
                contact_point1[index] = body_b_pos

                # convert n to world frame
                n = wp.transform_vector(X_ws_b, n)
                contact_normal[index] = n
                
                contact_shape0[index] = shape_a
                contact_shape1[index] = shape_b

                contact_thickness[index] = thickness

def collide(model, state, experimental_sdf_collision=False):

    # clear old count
    model.soft_contact_count.zero_()
    
    if (model.particle_count and model.shape_count):
        wp.launch(
            kernel=create_soft_contacts,
            dim=model.particle_count*model.shape_count,
            inputs=[
                model.particle_count,
                state.particle_q, 
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo_type, 
                model.shape_geo_id,
                model.shape_geo_scale,
                model.soft_contact_margin,
            ],
            outputs = [
                model.soft_contact_count,
                model.soft_contact_particle,
                model.soft_contact_body,
                model.soft_contact_body_pos,
                model.soft_contact_body_vel,
                model.soft_contact_normal,
                model.soft_contact_max
            ],
            device=model.device)

    # clear old count
    model.rigid_contact_count.zero_()

    if (True and model.body_count):
        wp.launch(
            kernel=broadphase_collision_pairs,
            dim=model.shape_contact_pair_count,
            inputs=[
                model.shape_contact_pairs,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo_type,
                model.shape_collision_radius,
                model.rigid_contact_max,
                model.mesh_num_points,
                model.rigid_contact_margin,
            ],
            outputs=[
                model.rigid_contact_count,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point_id,
            ],
            device=model.device)

        # print("rigid_contact_count:", model.rigid_contact_count.numpy()[0])
        # print("ground_contact_count:", ground_contact_count.numpy()[0])
            
        wp.launch(
            kernel=handle_contact_pairs,
            dim=model.rigid_contact_max,
            inputs=[
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo_type, 
                model.shape_geo_id,
                model.shape_geo_scale,
                model.shape_contact_thickness,
                model.rigid_contact_margin,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point_id,
                model.rigid_contact_count,
            ],
            outputs=[
                model.rigid_contact_body0,
                model.rigid_contact_body1,
                model.rigid_contact_point0,
                model.rigid_contact_point1,
                model.rigid_contact_offset0,
                model.rigid_contact_offset1,
                model.rigid_contact_normal,
                model.rigid_contact_thickness,
            ],
            device=model.device)
            
        if (model.ground):
            # print("Contacts before:", model.ground_contact_point0.numpy())
            # print(model.ground_contact_ref.numpy())
            wp.launch(
                kernel=update_rigid_ground_contacts,
                dim=model.ground_contact_dim,
                inputs=[
                    model.ground_plane,
                    model.ground_contact_body0,
                    state.body_q,
                    model.shape_transform,
                    model.ground_contact_ref,
                    model.ground_contact_shape0,
                    model.shape_contact_thickness,
                    model.rigid_contact_margin,
                    model.rigid_contact_max,
                ],
                outputs=[
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
                    model.rigid_contact_thickness,
                ],
                device=model.device
            )

        # print("rigid_contact_count:", state.rigid_contact_count.numpy()[0])
    
    # if False and experimental_sdf_collision:
    #     for shape_a in range(model.shape_count-1):
    #         point_count = model.mesh_num_points.numpy()[shape_a]
    #         for shape_b in range(shape_a+1, model.shape_count):
    #             # print(f'colliding {shape_a} {shape_b}')
    #             wp.launch(
    #                 kernel=create_mesh_sdf_contacts,
    #                 dim=point_count,
    #                 inputs=[
    #                     shape_a,
    #                     shape_b,
    #                     state.body_q,
    #                     model.shape_transform,
    #                     model.shape_body,
    #                     model.shape_geo_type, 
    #                     model.shape_geo_id,
    #                     model.shape_volume_id,
    #                     model.shape_geo_scale,
    #                     model.shape_contact_thickness,
    #                     model.rigid_contact_max,
    #                     model.rigid_contact_margin,
    #                 ],
    #                 # outputs
    #                 outputs=[
    #                     model.rigid_contact_count,
    #                     model.rigid_contact_body0,
    #                     model.rigid_contact_body1,
    #                     model.rigid_contact_point0,
    #                     model.rigid_contact_point1,
    #                     model.rigid_contact_offset0,
    #                     model.rigid_contact_offset1,
    #                     model.rigid_contact_normal,
    #                     model.rigid_contact_shape0,
    #                     model.rigid_contact_shape1,
    #                     model.rigid_contact_thickness,
    #                 ],
    #                 device=model.device)
