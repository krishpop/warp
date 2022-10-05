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




@wp.kernel
def create_soft_contacts(
    num_particles: int,
    particle_x: wp.array(dtype=wp.vec3), 
    body_X_sc: wp.array(dtype=wp.transform),
    shape_X_co: wp.array(dtype=wp.transform),
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

    X_sc = wp.transform_identity()
    if (rigid_index >= 0):
        X_sc = body_X_sc[rigid_index]
    
    X_co = shape_X_co[shape_index]

    X_so = wp.transform_multiply(X_sc, X_co)
    X_os = wp.transform_inverse(X_so)
    
    # transform particle position to shape local space
    x_local = wp.transform_point(X_os, px)

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
            body_pos = wp.transform_point(X_co, x_local - n*d)
            body_vel = wp.transform_vector(X_co, v)

            world_normal = wp.transform_vector(X_so, n)

            soft_contact_body[index] = rigid_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = body_vel
            soft_contact_particle[index] = particle_index
            soft_contact_normal[index] = world_normal


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


# @wp.kernel
# def create_mesh_sdf_contacts(
#     shape_a: int,
#     shape_b: int,
#     body_q: wp.array(dtype=wp.transform),
#     body_qd: wp.array(dtype=wp.spatial_vector),
#     shape_X_co: wp.array(dtype=wp.transform),
#     shape_body: wp.array(dtype=int),
#     shape_geo_type: wp.array(dtype=int), 
#     shape_geo_id: wp.array(dtype=wp.uint64),
#     shape_volume_id: wp.array(dtype=wp.uint64),
#     shape_geo_scale: wp.array(dtype=wp.vec3),
#     shape_materials: wp.array(dtype=wp.vec4),
#     rigid_contact_thickness: float,
#     #outputs,
#     rigid_contact_count: wp.array(dtype=int),
#     rigid_contact_body_a: wp.array(dtype=int),
#     rigid_contact_body_a_pos: wp.array(dtype=wp.vec3),
#     rigid_contact_body_a_vel: wp.array(dtype=wp.vec3),
#     rigid_contact_body_b: wp.array(dtype=int),
#     rigid_contact_body_b_pos: wp.array(dtype=wp.vec3),
#     rigid_contact_body_b_vel: wp.array(dtype=wp.vec3),
#     rigid_contact_normal: wp.array(dtype=wp.vec3),
#     rigid_contact_distance: wp.array(dtype=float),
#     rigid_contact_max: int,    
#     rigid_contact_material: wp.array(dtype=wp.vec4)):
    
#     tid = wp.tid()           

#     rigid_a = shape_body[shape_a]
#     rigid_b = shape_body[shape_b]

#     X_sc_a = body_q[rigid_a]
#     X_sc_b = body_q[rigid_b]
    
#     X_co_a = shape_X_co[shape_a]
#     X_co_b = shape_X_co[shape_b]

#     X_so_a = wp.transform_multiply(X_sc_a, X_co_a)
#     # X_os_a = wp.transform_inverse(X_so_a)
#     X_so_b = wp.transform_multiply(X_sc_b, X_co_b)


#     X_so_a = X_sc_a
#     X_so_b = X_sc_b


#     X_os_b = wp.transform_inverse(X_so_b)


    

#     # geo description
#     geo_type_a = shape_geo_type[shape_a]
#     geo_scale_a = shape_geo_scale[shape_a]
#     geo_type_b = shape_geo_type[shape_b]
#     geo_scale_b = shape_geo_scale[shape_b]

#     # evaluate shape sdf
#     d = 1.e+6 
#     n = wp.vec3()
#     v = wp.vec3()

#     # GEO_SPHERE (0)
#     # if (geo_type == 0):
#     #     d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
#     #     n = sphere_sdf_grad(wp.vec3(), geo_scale[0], x_local)

#     # # GEO_BOX (1)
#     # if (geo_type == 1):
#     #     d = box_sdf(geo_scale, x_local)
#     #     n = box_sdf_grad(geo_scale, x_local)
        
#     # # GEO_CAPSULE (2)
#     # if (geo_type == 2):
#     #     d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
#     #     n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local)

#     # # GEO_MESH (3)
#     if (geo_type_a == 3 and geo_type_b == 3):
#         # mesh vertex <> volume contact
#         mesh_a = shape_geo_id[shape_a]
#         mesh_b = shape_geo_id[shape_b]
#         # volume = shape_volume_id[shape_b]
#         # print(volume)

#         # print(mesh)
#         # print("mesh")
#         # print(mesh)
#         # print("point")
#         # print(tid)

#         body_a_pos = wp.mesh_get_point(mesh_a, tid) * geo_scale_a[0]
#         p_world = wp.transform_point(X_so_a, body_a_pos)

#         query_b_local = wp.transform_point(X_os_b, p_world)

#         # print("p_mesh")
#         # print(p_mesh)
#         # print("p_world")
#         # print(p_world)
#         # print("p_vol")
#         # print(p_vol)

        
#         # print("retrieved")

#             # # transform point to world space   
#             # p_vol_local = wp.volume_world_to_index(volume, p_vol) #* 0.2
#             # # p_vol_local = p_vol * 5.0

#             # # print("p_vol_local")
#             # # print(p_vol_local)
            
#             # # print("query volume")
#             # # print(volume)
#             # d = wp.volume_sample_f(volume, p_vol_local, wp.Volume.LINEAR)
#             # # print("query volume grad")
#             # n = volume_grad(volume, p_vol)
#             # # print("done")

#         # print("d")
#         # print(d)

#         # print(d)
#         face_index = int(0)
#         face_u = float(0.0)  
#         face_v = float(0.0)
#         sign = float(0.0)

#         if (wp.mesh_query_point(mesh_b, query_b_local/geo_scale_b[0], 1.0, sign, face_index, face_u, face_v)):

#             shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
#             shape_v = wp.mesh_eval_velocity(mesh_b, face_index, face_u, face_v)

#             shape_p = shape_p*geo_scale_b[0]
#             shape_v = shape_v*geo_scale_b[0]

#             delta = query_b_local-shape_p
#             d = wp.length(delta)*sign
#             n = wp.normalize(delta)*sign
#             v = shape_v

#             # print("query successful")
#             # print(d)

#         rigid_contact_thickness = 0.01


#         if (d < rigid_contact_thickness):

#             index = wp.atomic_add(rigid_contact_count, 0, 1) 
#             # print("contact")
#             # print(d)
#             # print(n)
#             # print(sign)

#             # print("rigids")
#             # print(rigid_a)
#             # print(rigid_b)

#             if (index < rigid_contact_max):

#                 # n = wp.transform_vector(X_so, n)
#                 err = d - rigid_contact_thickness

#                 # mesh collision
#                 # compute point at the surface of volume b
#                 body_b_pos = shape_p - n*err
#                 # xpred = xpred - n*d
#                 # body_b_pos = shape_p

#                 body_b_pos_world = wp.transform_point(X_so_b, body_b_pos)
#                 # body_a_pos = p_mesh # wp.transform_point(X_os_a, body_b_pos_world)

#                 # compute contact point in body local space
#                 # body_a_pos = wp.transform_point(X_co_a, xpred)
#                 body_a_vel = wp.transform_vector(X_co_a, v)


#                 rigid_contact_body_a[index] = rigid_a
#                 rigid_contact_body_a_pos[index] = body_a_pos
#                 rigid_contact_body_a_vel[index] = body_a_vel

#                 # TODO verify
#                 qd_b = body_qd[rigid_b]
#                 v_b = wp.spatial_bottom(qd_b)
#                 w_b = wp.spatial_top(qd_b)
#                 p_b = wp.transform_get_translation(X_sc_b)
#                 q_b = wp.transform_get_rotation(X_sc_b)
#                 b_vel = v_b + wp.cross(w_b, body_b_pos)
                
#                 rigid_contact_body_b[index] = rigid_b
#                 rigid_contact_body_b_pos[index] = body_b_pos
#                 rigid_contact_body_b_vel[index] = b_vel

#                 # convert n to world frame
#                 n = wp.transform_vector(X_so_b, n)
#                 # n = wp.transform_vector(X_os_b, n)
#                 # print(n)
#                 rigid_contact_normal[index] = n

#                 mat_a = shape_materials[shape_a]
#                 mat_b = shape_materials[shape_b]
#                 # XXX use average of both colliding materials
#                 rigid_contact_material[index] = 0.5 * (mat_a + mat_b)
#                 rigid_contact_distance[index] = d

@wp.kernel
def update_rigid_ground_contacts(
    ground_plane: wp.array(dtype=float),
    rigid_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_X_co: wp.array(dtype=wp.transform),
    contact_point_ref: wp.array(dtype=wp.vec3),
    ground_contact_shape: wp.array(dtype=int),
    shape_contact_thickness: wp.array(dtype=float),
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
    contact_thickness: wp.array(dtype=float)
):
    tid = wp.tid()
    body = rigid_body[tid]
    shape = ground_contact_shape[tid]
    thickness = shape_contact_thickness[shape]
    X_wb = body_q[body]
    X_bw = wp.transform_inverse(X_wb)
    X_co = shape_X_co[shape]
    X_ws = wp.transform_multiply(X_wb, X_co)
    n = wp.vec3(ground_plane[0], ground_plane[1], ground_plane[2])
    p_ref = wp.transform_point(X_ws, contact_point_ref[tid])
    c = ground_plane[3]  # ground plane offset
    d = wp.dot(p_ref, n) - c
    if (d < thickness + rigid_contact_margin):
        index = wp.atomic_add(contact_count, 0, 1)
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


@wp.kernel
def create_mesh_sdf_contacts(
    shape_a: int,
    shape_b: int,
    body_q: wp.array(dtype=wp.transform),
    shape_X_co: wp.array(dtype=wp.transform),
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

    X_sc_a = body_q[rigid_a]
    X_sc_b = body_q[rigid_b]
    
    X_co_a = shape_X_co[shape_a]
    X_co_b = shape_X_co[shape_b]

    X_so_a = wp.transform_multiply(X_sc_a, X_co_a)
    X_os_a = wp.transform_inverse(X_so_a)
    X_so_b = wp.transform_multiply(X_sc_b, X_co_b)
    X_os_b = wp.transform_inverse(X_so_b)

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
    # if (geo_type == 0):
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
        p_world = wp.transform_point(X_so_a, body_a_pos)

        # print("body_a_pos")
        # print(body_a_pos)
        # print(tid)

        query_b_local = wp.transform_point(X_os_b, p_world)

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
            index = wp.atomic_add(contact_count, 0, 1)

            if (index < contact_max):
                err = d - thickness

                # mesh collision
                # compute point at the surface of volume b
                body_b_pos = shape_p - n*err

                # offset by contact thickness to be used in PBD contact friction constraints
                contact_offset0[index] = wp.transform_vector(wp.transform_inverse(X_os_a), -thickness_a * n)
                contact_offset1[index] = wp.transform_vector(wp.transform_inverse(X_os_b), -thickness_b * n)

                # assign contact points in body local spaces
                contact_body0[index] = rigid_a
                contact_point0[index] = body_a_pos
                
                contact_body1[index] = rigid_b
                contact_point1[index] = body_b_pos

                # convert n to world frame
                n = wp.transform_vector(X_so_b, n)
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

    if (model.ground and model.body_count):

        if (hasattr(state, "rigid_contact_count")):
            contact_outputs = [
                state.rigid_contact_count,
                state.rigid_contact_body0,
                state.rigid_contact_body1,
                state.rigid_contact_point0,
                state.rigid_contact_point1,
                state.rigid_contact_offset0,
                state.rigid_contact_offset1,
                state.rigid_contact_normal,
                state.rigid_contact_shape0,
                state.rigid_contact_shape1,
                state.rigid_contact_thickness,
            ]
        else:
            contact_outputs = [
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
            ]
        # print("Contacts before:", model.ground_contact_point0.numpy())
        # print(model.ground_contact_ref.numpy())
        wp.launch(
            kernel=update_rigid_ground_contacts,
            dim=min(model.ground_contact_dim, model.rigid_contact_max),
            inputs=[
                model.ground_plane,
                model.ground_contact_body0,
                state.body_q,
                model.shape_transform,
                model.ground_contact_ref,
                model.ground_contact_shape0,
                model.shape_contact_thickness,
                model.rigid_contact_margin,
            ],
            outputs=contact_outputs,
            device=model.device
        )

        # print("rigid_contact_count:", state.rigid_contact_count.numpy()[0])

    
    if experimental_sdf_collision:
        for shape_a in range(model.shape_count-1):
            point_count = model.mesh_num_points[shape_a]
            for shape_b in range(shape_a+1, model.shape_count):
                # print(f'colliding {shape_a} {shape_b}')
                wp.launch(
                    kernel=create_mesh_sdf_contacts,
                    dim=point_count,
                    inputs=[
                        shape_a,
                        shape_b,
                        state.body_q,
                        model.shape_transform,
                        model.shape_body,
                        model.shape_geo_type, 
                        model.shape_geo_id,
                        model.shape_volume_id,
                        model.shape_geo_scale,
                        model.shape_contact_thickness,
                        model.rigid_contact_max,
                        model.rigid_contact_margin,
                    ],
                    # outputs
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
                    device=model.device)
