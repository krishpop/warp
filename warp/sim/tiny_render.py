# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import sys

import warp as wp
import warp.sim

import numpy as np

@wp.kernel
def update_vbo(
    shape_ids: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    instance_envs: wp.array(dtype=int),
    env_offsets: wp.array(dtype=wp.vec3),
    scaling: float,
    bodies_per_env: int,
    # outputs
    vbo_positions: wp.array(dtype=wp.vec4),
    vbo_orientations: wp.array(dtype=wp.quat)):

    tid = wp.tid()
    shape = shape_ids[tid]
    body = shape_body[shape]
    env = instance_envs[tid]
    X_ws = shape_transform[shape]
    if body >= 0:
        X_ws = body_q[body+env*bodies_per_env] * X_ws
    p = wp.transform_get_translation(X_ws)
    q = wp.transform_get_rotation(X_ws)
    p *= scaling
    p += env_offsets[env]
    vbo_positions[tid] = wp.vec4(p[0], p[1], p[2], 0.0)
    vbo_orientations[tid] = q

class TinyRenderer:
    
    def __init__(
        self,
        model: warp.sim.Model,
        title="Warp sim",
        scaling=1.0,
        fps=60,
        upaxis="y",
        env_offset=(5.0, 0.0, 5.0),
        suppress_keyboard_help=False,
        start_paused=False):

        import pytinyopengl3 as p
        self.p = p

        self.paused = start_paused
        self.skip_rendering = False

        self.app = p.TinyOpenGL3App(title)
        self.app.renderer.init()
        def keypress(key, pressed):
            if not pressed:
                return
            if key == 27:  # ESC
                self.app.window.set_request_exit()
                sys.exit(0)
            if key == 32:  # space
                self.paused = not self.paused
            if key == ord('s'):
                self.skip_rendering = not self.skip_rendering

        self.app.window.set_keyboard_callback(keypress)
        self.cam = p.TinyCamera()
        self.cam.set_camera_distance(15.)
        self.cam.set_camera_pitch(-30)
        self.cam.set_camera_yaw(45)
        self.cam.set_camera_target_position(0.0, 0.0, 0.0)
        self.cam_axis = "xyz".index(upaxis.lower())
        self.cam.set_camera_up_axis(self.cam_axis)
        self.app.renderer.set_camera(self.cam)

        self.model = model
        self.num_envs = model.num_envs
        self.shape_body = model.shape_body.numpy()
        self.shape_transform = model.shape_transform.numpy()

        self.scaling = scaling

        # mapping from visual index to simulation shape
        self.shape_ids = {}
        # mapping from instance to shape ID
        self.instance_shape = []

        # render meshes double sided
        double_sided_meshes = False

        # create rigid shape children
        if (self.model.shape_count):
            shape_body = model.shape_body.numpy()
            body_q = model.body_q.numpy()
            shape_geo_src = model.shape_geo_src#.numpy()
            shape_geo_type = model.shape_geo_type.numpy()
            shape_geo_scale = model.shape_geo_scale.numpy()
            shape_transform = model.shape_transform.numpy()
            
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab20')
            colors20 = [(np.array(cmap(i)[:3])*255).astype(int) for i in np.linspace(0, 1, 20)]

            for s in range((model.shape_count-1) // self.num_envs):
                geo_type = shape_geo_type[s]
                geo_scale = shape_geo_scale[s] * self.scaling
                geo_src = shape_geo_src[s]

                # shape transform in body frame
                body = shape_body[s]
                if body > -1:
                    X_ws = wp.transform_expand(wp.mul(body_q[body], shape_transform[s]))
                else:
                    X_ws = wp.transform_expand(shape_transform[s])
                scale = np.ones(3)

                if (geo_type == warp.sim.GEO_PLANE):
                    color1 = (np.array(colors20[s%20]) + 50.0).clip(0, 255).astype(int)
                    color2 = (np.array(colors20[s%20]) + 90.0).clip(0, 255).astype(int)
                    texture = self.create_check_texture(256, 256, color1=color1, color2=color2)
                    faces = [0, 1, 2, 2, 3, 0]
                    normal = (0.0, 1.0, 0.0)
                    width = (geo_scale[0] if geo_scale[0] > 0.0 else 100.0)
                    length = (geo_scale[1] if geo_scale[1] > 0.0 else 100.0)
                    aspect = width / length
                    u = width / scaling * aspect
                    v = length / scaling
                    gfx_vertices = [
                        -width, 0.0, -length, 0.0, *normal, 0.0, 0.0,
                        -width, 0.0,  length, 0.0, *normal, 0.0, v,
                         width, 0.0,  length, 0.0, *normal, u, v,
                         width, 0.0, -length, 0.0, *normal, u, 0.0,
                    ]
                    shape = self.app.renderer.register_shape(gfx_vertices, faces, texture, double_sided_meshes)

                elif (geo_type == warp.sim.GEO_SPHERE):
                    texture = self.create_check_texture(color1=colors20[s%20])
                    shape = self.app.register_graphics_unit_sphere_shape(p.EnumSphereLevelOfDetail.SPHERE_LOD_HIGH, texture)
                    scale *= float(geo_scale[0]) * 2.0  # diameter

                elif (geo_type == warp.sim.GEO_CAPSULE):
                    radius = float(geo_scale[0])
                    half_width = float(geo_scale[1])
                    up_axis = 0
                    texture = self.create_check_texture(color1=colors20[s%20])
                    shape = self.app.register_graphics_capsule_shape(radius, half_width, up_axis, texture)

                elif (geo_type == warp.sim.GEO_BOX):
                    texture = self.create_check_texture(color1=colors20[s%20])
                    shape = self.app.register_cube_shape(geo_scale[0], geo_scale[1], geo_scale[2], texture, 4)

                elif (geo_type == warp.sim.GEO_MESH):
                    texture = self.create_check_texture(1, 1, color1=colors20[s%20], color2=colors20[s%20])
                    faces = geo_src.indices.reshape((-1, 3))
                    vertices = np.array(geo_src.vertices)
                    # convert vertices to (x,y,z,w, nx,ny,nz, u,v) format
                    if False:
                        gfx_vertices = np.hstack((vertices * geo_scale, np.zeros((len(geo_src.vertices), 6))))
                        gfx_indices = faces[:, ::-1]
                    else:
                        gfx_vertices = np.zeros((len(faces)*3, 9))
                        gfx_indices = np.arange(len(faces)*3).reshape((-1, 3))
                        # compute vertex normals
                        for i, f in enumerate(faces):
                            v0 = vertices[f[0]] * geo_scale
                            v1 = vertices[f[1]] * geo_scale
                            v2 = vertices[f[2]] * geo_scale
                            gfx_vertices[i*3+0, :3] = v0
                            gfx_vertices[i*3+1, :3] = v1
                            gfx_vertices[i*3+2, :3] = v2
                            n = np.cross(v1-v0, v2-v0)
                            gfx_vertices[i*3:i*3+3, 4:7] = n / np.linalg.norm(n)
                        
                    shape = self.app.renderer.register_shape(
                        gfx_vertices.flatten(),
                        gfx_indices.flatten(),
                        texture,
                        double_sided_meshes)

                elif (geo_type == warp.sim.GEO_SDF):
                    continue
                else:
                    print("Unknown geometry type: ", geo_type)
                    continue
                instance_pos = [p.TinyVector3f(*X_ws.p)] * self.num_envs
                instance_orn = [p.TinyQuaternionf(*X_ws.q)] * self.num_envs
                instance_color = [p.TinyVector3f(1.,1.,1.)] * self.num_envs
                instance_scale = [p.TinyVector3f(*scale)] * self.num_envs
                opacity = 1
                rebuild = True
                self.shape_ids[shape] = s
                self.instance_shape.extend([s] * self.num_envs)
                self.app.renderer.register_graphics_instances(
                    shape, instance_pos, instance_orn,
                    instance_color, instance_scale, opacity, rebuild
                )

        if model.ground:
            color1 = (200, 200, 200)
            color2 = (150, 150, 150)
            texture = self.create_check_texture(256, 256, color1=color1, color2=color2)
            faces = [0, 1, 2, 2, 3, 0]
            normal = (0.0, 1.0, 0.0)
            geo_scale = shape_geo_scale[-1]
            width = 100.0 * scaling
            length = 100.0 * scaling
            u = 100.0
            v = 100.0
            gfx_vertices = [
                -width, 0.0, -length, 0.0, *normal, 0.0, 0.0,
                -width, 0.0,  length, 0.0, *normal, 0.0, v,
                 width, 0.0,  length, 0.0, *normal, u, v,
                 width, 0.0, -length, 0.0, *normal, u, 0.0,
            ]
            shape = self.app.renderer.register_shape(gfx_vertices, faces, texture, double_sided_meshes)
            X_ws = wp.transform_expand(shape_transform[-1])
            pos = p.TinyVector3f(*X_ws.p)
            orn = p.TinyQuaternionf(*X_ws.q)
            color = p.TinyVector3f(1.,1.,1.)
            scale = p.TinyVector3f(1.,1.,1.)
            opacity = 1
            rebuild = True
            self.app.renderer.register_graphics_instance(shape, pos, orn, color, scale, opacity, rebuild)

        self.app.renderer.write_transforms()
        
        self.num_shapes = len(self.shape_ids)
        self.num_instances = self.num_shapes * self.num_envs
        self.bodies_per_env = len(self.model.body_q) // self.num_envs
        
        # mapping from shape instance to environment ID
        self.instance_envs = wp.array(
            np.tile(np.arange(self.num_envs, dtype=np.int32), self.num_instances), dtype=wp.int32,
            device="cuda", owner=False, ndim=1)
        # compute offsets per environment
        nonzeros = np.nonzero(env_offset)[0]
        num_dim = nonzeros.shape[0]
        if num_dim > 0:
            side_length = int(np.ceil(self.num_envs**(1.0/num_dim)))
            self.env_offsets = []
        else:
            self.env_offsets = np.zeros((self.num_envs, 3))
        if num_dim == 1:
            for i in range(self.num_envs):
                self.env_offsets.append(i*env_offset)
        elif num_dim == 2:
            for i in range(self.num_envs):
                d0 = i // side_length
                d1 = i % side_length
                offset = np.zeros(3)
                offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
                offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
                self.env_offsets.append(offset)
        elif num_dim == 3:
            for i in range(self.num_envs):
                d0 = i // (side_length*side_length)
                d1 = (i // side_length) % side_length
                d2 = i % side_length
                offset = np.zeros(3)
                offset[0] = d0 * env_offset[0]
                offset[1] = d1 * env_offset[1]
                offset[2] = d2 * env_offset[2]
                self.env_offsets.append(offset)
        self.env_offsets = np.array(self.env_offsets)
        min_offsets = np.min(self.env_offsets, axis=0)
        correction = min_offsets + (np.max(self.env_offsets, axis=0) - min_offsets) / 2.0
        correction[self.cam_axis] = 0.0  # ensure the envs are not shifted below the ground plane
        self.env_offsets -= correction
        self.env_offsets = wp.array(self.env_offsets, dtype=wp.vec3, device="cuda")
        self.instance_shape = wp.array(self.instance_shape, dtype=wp.int32, device="cuda")
        # make sure the static arrays are on the GPU
        if self.model.shape_transform.device.is_cuda:
            self.shape_transform = self.model.shape_transform
            self.shape_body = self.model.shape_body
        else:
            self.shape_transform = self.model.shape_transform.to("cuda")
            self.shape_body = self.model.shape_body.to("cuda")

        # load VBO for direct access to shape instance transforms on GPU
        self.vbo = self.app.cuda_map_vbo()
        self.vbo_positions = wp.array(
            ptr=self.vbo.positions, dtype=wp.vec4, shape=(self.num_instances,),
            length=self.num_instances, capacity=self.num_instances,
            device="cuda", owner=False, ndim=1)
        self.vbo_orientations = wp.array(
            ptr=self.vbo.orientations, dtype=wp.quat, shape=(self.num_instances,),
            length=self.num_instances, capacity=self.num_instances,
            device="cuda", owner=False, ndim=1)
        self._graph = None  # for CUDA graph recording

        if not suppress_keyboard_help:
            print("Keyboard commands for the TinyRenderer window:")
            print("  [Space] - pause simulation")
            print("  [S]     - skip rendering")
            print("  [ESC]   - exit")

    def __del__(self):
        self.app.cuda_unmap_vbo()

    def render(self, state: warp.sim.State):
        if self.skip_rendering:
            return

        if (self.model.particle_count):
            pass

            # particle_q = state.particle_q.numpy()

            # # render particles
            # self.render_points("particles", particle_q, radius=self.model.soft_contact_distance)

            # # render tris
            # if (self.model.tri_count):
            #     self.render_mesh("surface", particle_q, self.model.tri_indices.numpy().flatten())

            # # render springs
            # if (self.model.spring_count):
            #     self.render_line_list("springs", particle_q, self.model.spring_indices.numpy().flatten(), [], 0.1)

        # render muscles
        if (self.model.muscle_count):
            pass
            
            # body_q = state.body_q.numpy()

            # muscle_start = self.model.muscle_start.numpy()
            # muscle_links = self.model.muscle_bodies.numpy()
            # muscle_points = self.model.muscle_points.numpy()
            # muscle_activation = self.model.muscle_activation.numpy()

            # for m in range(self.model.muscle_count):

            #     start = int(muscle_start[m])
            #     end = int(muscle_start[m + 1])

            #     points = []

            #     for w in range(start, end):
                    
            #         link = muscle_links[w]
            #         point = muscle_points[w]

            #         X_sc = wp.transform_expand(body_q[link][0])

            #         points.append(Gf.Vec3f(wp.transform_point(X_sc, point).tolist()))
                
            #     self.render_line_strip(name=f"muscle_{m}", vertices=points, radius=0.0075, color=(muscle_activation[m], 0.2, 0.5))
        
        

        # update bodies
        if (self.model.body_count):
            
            wp.synchronize()
            if state.body_q.device.is_cuda:
                self.body_q = state.body_q
            else:
                self.body_q = state.body_q.to("cuda")

            if self._graph is None:
                wp.capture_begin()
                wp.launch(
                    update_vbo,
                    dim=len(self.instance_shape),
                    inputs=[
                        self.instance_shape,
                        self.shape_body,
                        self.shape_transform,
                        self.body_q,
                        self.instance_envs,
                        self.env_offsets,
                        self.scaling,
                        self.bodies_per_env,
                    ],
                    outputs=[
                        self.vbo_positions,
                        self.vbo_orientations,
                    ],
                    device="cuda")
                self._graph = wp.capture_end()
            else:
                wp.capture_launch(self._graph)

    def begin_frame(self, time: float):
        self.time = time
        while self.paused and not self.app.window.requested_exit():
            self.update()
        if self.app.window.requested_exit():
            sys.exit(0)

    def end_frame(self):
        self.update()

    def update(self):
        if self.skip_rendering:
            # ensure we receive key events
            self.app.swap_buffer()
            return
        self.app.renderer.update_camera(self.cam_axis)
        self.app.renderer.render_scene()
        self.app.swap_buffer()

    def save(self):
        while not self.app.window.requested_exit():
            self.update()
        if self.app.window.requested_exit():
            sys.exit(0)

    def create_check_texture(self, width=256, height=256, color1=(0, 128, 256), color2=(255, 255, 255)):
        pixels = np.zeros((width, height, 3), dtype=np.uint8)
        half_w = width // 2
        half_h = height // 2
        pixels[0:half_w, 0:half_h] = color1
        pixels[half_w:width, half_h:height] = color1
        pixels[half_w:width, 0:half_h] = color2
        pixels[0:half_w, half_h:height] = color2
        return self.app.renderer.register_texture(pixels.flatten().tolist(), width, height, False)