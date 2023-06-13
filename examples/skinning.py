# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Creates skinning information for skeletal animation using heat diffusion

# Usage:

# --------------------------------------------------------------------------------
# compute_skinning_info(bone_transforms, bone_shapes, vertices, tri_ids, max_bones_per_vertex, resolution, device)
# --------------------------------------------------------------------------------
#
# bone_transforms: array of wp.mat44 describing the poses of the bones
# bone_shapes: array of wp.vec3 describing the local shape of the bone 
# vertices: array of wp.vec3 describing the mesh vertices
# tri_ids: array of int describing the mesh triangle indices
# max_bones_per_vertex: int describes the maximum number of bones a vertex can reference
# resolution of the grid to compute the weights
# device, either "cpu" or "cuda"

# Returned arrays:
# 
# bone_indices: array size len(vertices) * max_bones_per_vertex ints describing the bones references by the vertices
# bone_weights: array size len(vertices) * max_bones_per_vertex float describing the bones references by the vertices

import os
import math
import numpy as np
import warp as wp

default_device = "cuda"


@wp.func
def intersects_box(tri_p0: wp.vec3, tri_p1: wp.vec3, tri_p2: wp.vec3, box_center: wp.vec3, box_extents: wp.vec3):

    v0 = tri_p0 - box_center
    v1 = tri_p1 - box_center
    v2 = tri_p2 - box_center
    f0 = tri_p1 - tri_p0
    f1 = tri_p2 - tri_p1
    f2 = tri_p0 - tri_p2
    
    n = wp.cross(f0, f1)
    d = wp.dot(n, v0)
    r = box_extents[0] * abs(n[0]) + box_extents[1] * abs(n[1]) + box_extents[2] * abs(n[2])
    if d > r or d < -r:
        return False
    
    if wp.max(wp.max(v0[0], v1[0]), v2[0]) < -box_extents[0] or wp.min(wp.min(v0[0], v1[0]), v2[0]) > box_extents[0]:
        return False

    if wp.max(wp.max(v0[1], v1[1]), v2[1]) < -box_extents[1] or wp.min(wp.min(v0[1], v1[1]), v2[1]) > box_extents[1]:
        return False

    if wp.max(wp.max(v0[2], v1[2]), v2[2]) < -box_extents[2] or wp.min(wp.min(v0[2], v1[2]), v2[2]) > box_extents[2]:
        return False
    
    a00 = wp.vec3(0.0, -f0[2], f0[1])
    p0 = wp.dot(v0, a00)
    p1 = wp.dot(v1, a00)
    p2 = wp.dot(v2, a00)
    r = box_extents[1] * abs(f0[2]) + box_extents[2] * abs(f0[1])
    if (wp.max(-wp.max(wp.max(p0, p1), p2), wp.min(wp.min(p0, p1), p2))) > r:
        return False

    a01 = wp.vec3(0.0, -f1[2], f1[1])
    p0 = wp.dot(v0, a01)
    p1 = wp.dot(v1, a01)
    p2 = wp.dot(v2, a01)
    r = box_extents[1] * abs(f1[2]) + box_extents[2] * abs(f1[1])
    if (wp.max(-wp.max(wp.max(p0, p1), p2), wp.min(wp.min(p0, p1), p2))) > r:
        return False

    a02 = wp.vec3(0.0, -f2[2], f2[1])
    p0 = wp.dot(v0, a02)
    p1 = wp.dot(v1, a02)
    p2 = wp.dot(v2, a02)
    r = box_extents[1] * abs(f2[2]) + box_extents[2] * abs(f2[1])
    if (wp.max(-wp.max(wp.max(p0, p1), p2), wp.min(wp.min(p0, p1), p2))) > r:
        return False

    a10 = wp.vec3(f0[2], 0.0, -f0[0])
    p0 = wp.dot(v0, a10)
    p1 = wp.dot(v1, a10)
    p2 = wp.dot(v2, a10)
    r = box_extents[0] * abs(f0[2]) + box_extents[2] * abs(f0[0])
    if (wp.max(-wp.max(wp.max(p0, p1), p2), wp.min(wp.min(p0, p1), p2))) > r:
        return False

    a11 = wp.vec3(f1[2], 0.0, -f1[0])
    p0 = wp.dot(v0, a11)
    p1 = wp.dot(v1, a11)
    p2 = wp.dot(v2, a11)
    r = box_extents[0] * abs(f1[2]) + box_extents[2] * abs(f1[0])
    if (wp.max(-wp.max(wp.max(p0, p1), p2), wp.min(wp.min(p0, p1), p2))) > r:
        return False

    a11 = wp.vec3(f2[2], 0.0, -f2[0])
    p0 = wp.dot(v0, a11)
    p1 = wp.dot(v1, a11)
    p2 = wp.dot(v2, a11)
    r = box_extents[0] * abs(f2[2]) + box_extents[2] * abs(f2[0])
    if (wp.max(-wp.max(wp.max(p0, p1), p2), wp.min(wp.min(p0, p1), p2))) > r:
        return False

    a20 = wp.vec3(-f0[1], f0[0], 0.0)
    p0 = wp.dot(v0, a20)
    p1 = wp.dot(v1, a20)
    p2 = wp.dot(v2, a20)
    r = box_extents[0] * abs(f0[1]) + box_extents[1] * abs(f0[0])
    if (wp.max(-wp.max(wp.max(p0, p1), p2), wp.min(wp.min(p0, p1), p2))) > r:
        return False

    a21 = wp.vec3(-f1[1], f1[0], 0.0)
    p0 = wp.dot(v0, a21)
    p1 = wp.dot(v1, a21)
    p2 = wp.dot(v2, a21)
    r = box_extents[0] * abs(f1[1]) + box_extents[1] * abs(f1[0])
    if (wp.max(-wp.max(wp.max(p0, p1), p2), wp.min(wp.min(p0, p1), p2))) > r:
        return False

    a22 = wp.vec3(-f2[1], f2[0], 0.0)
    p0 = wp.dot(v0, a22)
    p1 = wp.dot(v1, a22)
    p2 = wp.dot(v2, a22)
    r = box_extents[0] * abs(f2[1]) + box_extents[1] * abs(f2[0])
    if (wp.max(-wp.max(wp.max(p0, p1), p2), wp.min(wp.min(p0, p1), p2))) > r:
        return False

    return True


@wp.kernel
def mark_intersected_cells_and_faces(
        verts: wp.array(dtype=wp.vec3),
        tri_ids: wp.array(dtype=int),
        cell_marks: wp.array(dtype=int),
        face_connected: wp.array(dtype=int),
        grid_orig: wp.vec3,
        cell_size: float,        
        num_x: int, num_y: int, num_z: int):

    tri_nr = wp.tid()
    p0 = verts[tri_ids[3*tri_nr]] - grid_orig
    p1 = verts[tri_ids[3*tri_nr + 1]] - grid_orig
    p2 = verts[tri_ids[3*tri_nr + 2]] - grid_orig

    x0 = wp.max(int(wp.floor(wp.min(wp.min(p0[0], p1[0]), p2[0]) / cell_size)), 0)
    y0 = wp.max(int(wp.floor(wp.min(wp.min(p0[1], p1[1]), p2[1]) / cell_size)), 0)
    z0 = wp.max(int(wp.floor(wp.min(wp.min(p0[2], p1[2]), p2[2]) / cell_size)), 0)

    x1 = wp.min(int(wp.floor(wp.max(wp.max(p0[0], p1[0]), p2[0]) / cell_size)), num_x - 1)
    y1 = wp.min(int(wp.floor(wp.max(wp.max(p0[1], p1[1]), p2[1]) / cell_size)), num_y - 1)
    z1 = wp.min(int(wp.floor(wp.max(wp.max(p0[2], p1[2]), p2[2]) / cell_size)), num_z - 1)

    box_extents = wp.vec3(cell_size, cell_size, cell_size) * 0.5               
    eps = 0.0001
    facex_extents = wp.vec3(eps, cell_size, cell_size) * 0.5
    facey_extents = wp.vec3(cell_size, eps, cell_size) * 0.5
    facez_extents = wp.vec3(cell_size, cell_size, eps) * 0.5
    
    for xi in range(x0, x1 + 1):
        for yi in range(y0, y1 + 1):
            for zi in range(z0, z1 + 1):
                # cell
                box_center = wp.vec3(float(xi) + 0.5, float(yi) + 0.5, float(zi) + 0.5) * cell_size
                cell_nr = (xi*num_y + yi)*num_z + zi
    
                if intersects_box(p0, p1, p2, box_center, box_extents):
                    cell_marks[cell_nr] = 1

                # cell faces
                facex_center = wp.vec3(float(xi) + 1.0, float(yi) + 0.5, float(zi) + 0.5) * cell_size 
                facey_center = wp.vec3(float(xi) + 0.5, float(yi) + 1.0, float(zi) + 0.5) * cell_size 
                facez_center = wp.vec3(float(xi) + 0.5, float(yi) + 0.5, float(zi) + 1.0) * cell_size 
                if intersects_box(p0, p1, p2, facex_center, facex_extents):
                    face_connected[3*cell_nr] = 1
                if intersects_box(p0, p1, p2, facey_center, facey_extents):
                    face_connected[3*cell_nr+1] = 1
                if intersects_box(p0, p1, p2, facez_center, facez_extents):
                    face_connected[3*cell_nr+2] = 1


@wp.kernel
def mark_boundary(
        cell_marks: wp.array(dtype=int),
        num_x: int, num_y: int, num_z: int):

    cell_nr = wp.tid()
    zi = cell_nr % num_z
    yi = int(cell_nr / num_z) % num_y
    xi = int(cell_nr / num_z) / num_y

    if xi == 0 or xi == num_x - 1 or yi == 0 or yi == num_y - 1 or zi == 0 or zi == num_z - 1:
        cell_marks[(xi * num_y + yi) * num_z + zi] = 2


@wp.kernel
def flood_outside(
        depth: int,
        cell_marks: wp.array(dtype=int),
        cell_updates: wp.array(dtype=int),
        num_x: int, num_y: int, num_z: int):

    cell_nr = wp.tid()
    mark = cell_marks[cell_nr]

    if mark != depth:
        return

    zi = cell_nr % num_z
    yi = int(cell_nr / num_z) % num_y
    xi = int(cell_nr / num_z) / num_y

    for i in range(6):
        nx = xi
        ny = yi
        nz = zi
        if i == 0:
            nx = wp.max(nx - 1, 0)
        elif i == 1:
            nx = wp.min(nx + 1, num_x - 1)
        elif i == 2:
            ny = wp.max(ny - 1, 0)
        elif i == 3:
            ny = wp.min(ny + 1, num_y - 1)
        elif i == 4:
            nz = wp.max(nz - 1, 0)
        elif i == 5:
            nz = wp.min(nz + 1, num_z - 1)
        n = (nx * num_y + ny) * num_z + nz

        if cell_marks[n] == 0:
            cell_marks[n] = depth + 1
            cell_updates[cell_nr] = 1
        

@wp.kernel
def init_heat_map(
        heat_map: wp.array(dtype=float),
        bone_marks: wp.array(dtype=int),
        num_x: int, num_y: int, num_z: int,
        grid_orig: wp.vec3,
        cell_size: float,
        bone_transforms: wp.array(dtype=wp.mat44),
        bone_shapes: wp.array(dtype=wp.vec3),
        num_bones: int,
        bone_nr: int):
    
    cell_nr = wp.tid()
    zi = cell_nr % num_z
    yi = int(cell_nr / num_z) % num_y
    xi = int(cell_nr / num_z) / num_y
    box_center = wp.vec3(float(xi) + 0.5, float(yi) + 0.5, float(zi) + 0.5) * cell_size
    box_extents = wp.vec3(cell_size, cell_size, cell_size) * 0.5      

    for i in range(num_bones):
        v = wp.mul(bone_transforms[i], wp.vec4(0.0, 0.0, 0.0, 1.0))
        p0 = wp.vec3(v[0], v[1], v[2]) - grid_orig
        s = bone_shapes[i]
        v = wp.mul(bone_transforms[i], wp.vec4(s[0], s[1], s[2], 1.0)) 
        p1 = wp.vec3(v[0], v[1], v[2]) - grid_orig
        
        if intersects_box(p0, p1, p1, box_center, box_extents):
            bone_marks[cell_nr] = 1
            if i == bone_nr:
                heat_map[cell_nr] = 1.0
            else:
                heat_map[cell_nr] = 0.0


@wp.kernel
def diffuse_heat_map(
        heat_map: wp.array(dtype=float),
        bone_marks: wp.array(dtype=int),
        cell_marks: wp.array(dtype=int),
        face_connected: wp.array(dtype=int),
        num_x: int, num_y: int, num_z: int):
    
    cell_nr = wp.tid()
    zi = cell_nr % num_z
    yi = int(cell_nr / num_z) % num_y
    xi = int(cell_nr / num_z) / num_y

    if cell_marks[cell_nr] > 1 or bone_marks[cell_nr] == 1:
        return
    
    sum = float(0.0)
    num = float(0.0)

    if xi > 0:
        adj = cell_nr - num_y * num_z
        if face_connected[3 * adj] == 1 and cell_marks[adj] <= 1:
            sum = sum + heat_map[adj]
            num = num + 1.0

    if xi < num_x - 1:
        adj = cell_nr + num_y * num_z
        if face_connected[3 * cell_nr] == 1 and cell_marks[adj] <= 1:
            sum = sum + heat_map[adj]
            num = num + 1.0

    if yi > 0:
        adj = cell_nr - num_z
        if face_connected[3 * adj + 1] == 1 and cell_marks[adj] <= 1:
            sum = sum + heat_map[adj]
            num = num + 1.0

    if yi < num_y - 1:
        adj = cell_nr + num_z
        if face_connected[3 * cell_nr + 1] == 1 and cell_marks[adj] <= 1:
            sum = sum + heat_map[adj]
            num = num + 1.0

    if zi > 0:
        adj = cell_nr - 1
        if face_connected[3 * adj + 2] == 1 and cell_marks[adj] <= 1:
            sum = sum + heat_map[adj]
            num = num + 1.0

    if zi < num_z - 1:
        adj = cell_nr + 1
        if face_connected[3 * cell_nr + 2] == 1 and cell_marks[adj] <= 1:
            sum = sum + heat_map[adj]
            num = num + 1.0

    # if num > 0.0:
    #     heat_map[cell_nr] = sum / num

    heat_map[cell_nr] = (sum + heat_map[cell_nr]) / (num + 1.0)

    
@wp.kernel
def update_skinning_info(
        verts: wp.array(dtype=wp.vec3),
        heat_map: wp.array(dtype=float),
        bone_nr: int,
        grid_orig: wp.vec3,
        cell_size: float, 
        num_x: int, num_y: int, num_z: int,
        skinning_ids: wp.array(dtype=int),
        skinning_weights: wp.array(dtype=float),
        max_bones_per_vertex: int):
    
    vert_nr = wp.tid()
    p = verts[vert_nr] - grid_orig

    xi = int(p[0] / cell_size)
    yi = int(p[1] / cell_size)
    zi = int(p[2] / cell_size)

    w = heat_map[(xi * num_y + yi) * num_z + zi]
    if w == 0.0:
        return

    first = vert_nr * max_bones_per_vertex

    pos = max_bones_per_vertex - 1
    while pos >= 0 and w > skinning_weights[first + pos]:
        if pos < max_bones_per_vertex - 1:
            skinning_weights[first + pos + 1] = skinning_weights[pos]
            skinning_ids[first + pos + 1] = skinning_ids[first + pos]
        pos = pos - 1

    pos = pos + 1
    if pos < max_bones_per_vertex:
        skinning_weights[first + pos] = w
        skinning_ids[first + pos] = bone_nr


@wp.kernel
def normalize_skinning_info(
        skinning_weights: wp.array(dtype=float),
        max_bones_per_vertex: int):
    
    vert_nr = wp.tid()
    first = vert_nr * max_bones_per_vertex

    s = float(0.0)
    for i in range(max_bones_per_vertex):
        s = s + skinning_weights[first + i]
    
    if s > 0.0:
        s = 1.0 / s
        for i in range(max_bones_per_vertex):
            skinning_weights[first + i] = skinning_weights[first + i] * s


def compute_skinning_info(bone_transforms, bone_shapes, vertices, tri_ids, max_bones_per_vertex, resolution, device):

    # create dense grid

    bounds_min = np.amin(vertices, 0)
    bounds_max = np.amax(vertices, 0)
    cell_size = np.max(bounds_max - bounds_min) / resolution

    grid_orig = wp.vec3(bounds_min[0] - 2.1 * cell_size, bounds_min[1] - 2.1 * cell_size, bounds_min[2] - 2.1 * cell_size)

    num_x = int((bounds_max[0] - grid_orig[0]) / cell_size) + 2
    num_y = int((bounds_max[1] - grid_orig[1]) / cell_size) + 2
    num_z = int((bounds_max[2] - grid_orig[2]) / cell_size) + 2

    num_cells = num_x * num_y * num_z
    num_tris = int(len(tri_ids) / 3)
    num_verts = len(vertices)
    
    # mark all cells and faces intersected by the input mesh

    verts = wp.array(vertices, dtype=wp.vec3, device=device)
    tri_ids = wp.array(tri_ids, dtype=int, device=device)

    cell_marks = wp.zeros(shape=(num_cells), dtype=int, device=device)
    face_connected = wp.zeros(shape=(3*num_cells), dtype=int, device=device)

    wp.launch(kernel = mark_intersected_cells_and_faces,
        inputs = [verts, tri_ids, cell_marks, face_connected, grid_orig, cell_size, num_x, num_y, num_z],
        dim = num_tris, device=device)
        
    # flood outside

    wp.launch(kernel = mark_boundary,
        inputs = [cell_marks, num_x, num_y, num_z],
        dim = num_cells, device=device)

    cell_updates = wp.zeros(shape=(num_cells), dtype=int, device=device)
    host_cell_updates = wp.zeros(shape=(num_cells), dtype=int, device="cpu")

    any_updates = True
    depth = 2

    while any_updates:

        cell_updates.zero_()
        wp.launch(kernel = flood_outside,
            inputs = [depth, cell_marks, cell_updates, num_x, num_y, num_z],
            dim = num_cells, device=device)

        wp.copy(host_cell_updates, cell_updates)
        if np.max(host_cell_updates.numpy()) == 0:
            any_updates = False
        
        depth += 1

    # create skinning info

    num_bones = len(bone_transforms)

    heat_map = wp.zeros(shape=(num_cells), dtype=float, device=device)
    bone_marks = wp.zeros(shape=(num_cells), dtype=int, device=device)
    bone_transforms = wp.array(bone_transforms, dtype=wp.mat44, device=device)
    bone_shapes = wp.array(bone_shapes, dtype=wp.vec3, device=device)

    skinning_ids = wp.zeros(shape=(num_verts * max_bones_per_vertex), dtype=int, device=device)
    skinning_weights = wp.zeros(shape=(num_verts * max_bones_per_vertex), dtype=float, device=device)

    num_iters = resolution // 2

    for bone_nr in range(num_bones):

        wp.launch(kernel = init_heat_map,
            inputs = [heat_map, bone_marks, num_x, num_y, num_z, grid_orig, cell_size, bone_transforms, bone_shapes, num_bones, bone_nr],
            dim = num_cells, device=device)
         
        for iter in range(num_iters):
            wp.launch(kernel = diffuse_heat_map,
                inputs = [heat_map, bone_marks, cell_marks, face_connected, num_x, num_y, num_z],
                dim = num_cells, device=device)
         
        wp.launch(kernel = update_skinning_info,
            inputs = [verts, heat_map, bone_nr, grid_orig, cell_size, num_x, num_y, num_z, skinning_ids, skinning_weights, max_bones_per_vertex],
            dim = num_verts, device=device)

    wp.launch(kernel = normalize_skinning_info,
        inputs = [skinning_weights, max_bones_per_vertex],
        dim = num_verts, device=device)

    return skinning_ids, skinning_weights


    # ------------------------ test -------------------------------------


def read_obj(filename):

    vertices = []
    face_ids = []
    tri_ids = []

    bone_mats = []
    bone_shapes = []

    for line in open(filename, "r"):

        values = line.split()
        if not values: 
            continue
        if values[0] == 'v':
            vertices.append([float(i) for i in values[1:4]])
        elif values[0] == 'f':

            face_size = len(values) - 1
            face_ids = []
            for v in values[1:]:
                w = v.split('/')
                id = int(w[0]) - 1
                face_ids.append(id)
            for i in range(1, face_size-1):
                tri_ids.append(face_ids[0])
                tri_ids.append(face_ids[i])
                tri_ids.append(face_ids[i + 1])
        elif values[0] == "bb":
            m = [float(i) for i in values[1:17]]
            bone_mats.append([
                [m[0], m[1], m[2], m[3]], 
                [m[4], m[5], m[6], m[7]], 
                [m[8], m[9], m[10], m[11]],
                [m[12], m[13], m[14], m[15]]])
        elif values[0] == "bs":
            bone_shapes.append([float(i) for i in values[1:4]])

    return bone_mats, bone_shapes, vertices, tri_ids


def test():

    filename = os.path.join(os.path.dirname(__file__), "assets", "mano_hand", "mano_watertight.obj")
    device = "cuda"

    bone_mats, bone_shapes, vertices, tri_ids = read_obj(filename)

    max_bones_per_vertex = 4
    resolution = 50

    skinning_ids, skinning_weights = compute_skinning_info(bone_mats, bone_shapes, vertices, tri_ids, 
                                                           max_bones_per_vertex, resolution, device)
    
    print("skinning done")
    

wp.init()

test()

    