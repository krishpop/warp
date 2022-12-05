# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math

import warp as wp
import warp.sim

import numpy as np


class TinyRenderer:
    
    def __init__(self, model: warp.sim.Model, title="Warp sim", scaling=1.0, fps=60, upaxis="y"):

        import pytinyopengl3 as p
        self.p = p

        self.app = p.TinyOpenGL3App(title)
        self.app.renderer.init()
        self.cam = p.TinyCamera()
        self.cam.set_camera_distance(15.)
        self.cam.set_camera_pitch(-30)
        self.cam.set_camera_yaw(-20)
        self.cam.set_camera_target_position(0.0, 0.0, 0.0)
        self.cam_axis = "xyz".index(upaxis.lower())
        self.cam.set_camera_up_axis(self.cam_axis)
        self.app.renderer.set_camera(self.cam)

        self.model = model
        self.shape_body = model.shape_body.numpy()
        self.shape_transform = model.shape_transform.numpy()

        self.scaling = scaling

        # mapping from visual index to simulation shape
        self.shape_ids = {}

        self.instance_ids = {}

        # render meshes double sided
        double_sided_meshes = True
        

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

            for s in range(model.shape_count):

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
                    if (s == model.shape_count-1 and not model.ground):
                        continue  # hide ground plane

                    color1 = (np.array(colors20[s%20]) + 50.0).clip(0, 255).astype(int)
                    color2 = (np.array(colors20[s%20]) + 90.0).clip(0, 255).astype(int)
                    texture = self.create_check_texture(256, 256, color1=color1, color2=color2)
                    faces = [0, 1, 2, 2, 3, 0]
                    normal = (0.0, 1.0, 0.0)
                    width = (geo_scale[0] if geo_scale[0] > 0.0 else 100.0) * scaling
                    length = (geo_scale[1] if geo_scale[1] > 0.0 else 100.0) * scaling
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
                    radius = float(geo_scale[0]) * scaling
                    half_width = float(geo_scale[1]) * scaling
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
                pos = p.TinyVector3f(*X_ws.p)
                orn = p.TinyQuaternionf(*X_ws.q)
                color = p.TinyVector3f(1.,1.,1.)
                scale = p.TinyVector3f(*scale)
                opacity = 1
                rebuild = True
                i = self.app.renderer.register_graphics_instance(shape, pos, orn, color, scale, opacity, rebuild)
                self.shape_ids[shape] = s
                self.instance_ids[i] = s
        
        self.app.renderer.write_transforms()


    def render(self, state: warp.sim.State):
        self.app.renderer.update_camera(self.cam_axis)


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
        
        

        # update  bodies
        if (self.model.body_count):
            body_q = state.body_q.numpy()

            for instance, shape in self.instance_ids.items():
                body = self.shape_body[shape]
                X_bs = self.shape_transform[shape]
                if body > -1:
                    X_wb = body_q[body]
                    X_ws = wp.mul(X_wb, X_bs)
                else:
                    X_ws = wp.transform_expand(X_bs)
                self.update_pose(instance, X_ws)

        self.app.renderer.write_transforms()  
        self.app.renderer.render_scene()
        self.app.swap_buffer()

    def update_pose(self, instance_id, pose: wp.transform):
        self.app.renderer.write_single_instance_transform_to_cpu(
            self.p.TinyVector3f(pose.p[0] * self.scaling, pose.p[1] * self.scaling, pose.p[2] * self.scaling),
            self.p.TinyQuaternionf(*pose.q),
            instance_id
        )

    def begin_frame(self, time: float):
        self.time = time

    def end_frame(self):
        self.app.renderer.render_scene()
        self.app.swap_buffer()

    def save(self):
        while not self.app.window.requested_exit():
            self.app.renderer.update_camera(self.cam_axis)
            self.app.renderer.render_scene()
            self.app.swap_buffer()

    def create_check_texture(self, width=256, height=256, color1=(0, 128, 256), color2=(255, 255, 255)):
        pixels = np.zeros((width, height, 3), dtype=np.uint8)
        half_w = width // 2
        half_h = height // 2
        pixels[0:half_w, 0:half_h] = color1
        pixels[half_w:width, half_h:height] = color1
        pixels[half_w:width, 0:half_h] = color2
        pixels[0:half_w, half_h:height] = color2
        return self.app.renderer.register_texture(pixels.flatten().tolist(), width, height, False)