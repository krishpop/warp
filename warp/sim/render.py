# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math

import warp as wp
import warp.sim
import warp.render

from collections import defaultdict
from typing import Union

import numpy as np

from warp.render.utils import solidify_mesh, tab10_color_map


def AbstractSimRenderer(renderer):
    class SimRenderer_t(renderer):

        use_unique_colors = True
        
        def __init__(self, model: warp.sim.Model, path, scaling=1.0, fps=60, upaxis="y"):
            # create USD stage
            super().__init__(path, scaling=scaling, fps=fps, upaxis=upaxis)

            self.skip_rendering = False

            self.model = model
            self.num_envs = model.num_envs
            self.body_names = []

            self.cam_axis = "xyz".index(upaxis.lower())

            self.body_env = []  # mapping from body index to its environment index
            env_id = 0
            self.bodies_per_env = model.body_count // self.num_envs
            # create rigid body nodes
            for b in range(model.body_count):
                body_name = f"body_{b}_{self.model.body_name[b].replace(' ', '_')}"
                self.body_names.append(body_name)
                self.register_body(body_name)
                self.body_env.append(env_id)
                if b > 0 and b % self.bodies_per_env == 0:
                    env_id += 1

            # create rigid shape children
            if (self.model.shape_count):
                # mapping from hash of geometry to shape ID
                self.geo_shape = {}

                self.instance_count = 0

                self.body_name = {}  # mapping from body name to its body ID
                self.body_shapes = defaultdict(list)  # mapping from body index to its shape IDs
                
                shape_body = model.shape_body.numpy()
                shape_geo_src = model.shape_geo_src
                shape_geo_type = model.shape_geo.type.numpy()
                shape_geo_scale = model.shape_geo.scale.numpy()
                shape_geo_thickness = model.shape_geo.thickness.numpy()
                shape_geo_is_solid = model.shape_geo.is_solid.numpy()
                shape_transform = model.shape_transform.numpy()

                p = np.zeros(3, dtype=np.float32)
                q = np.array([0., 0., 0., 1.], dtype=np.float32)
                scale = np.ones(3)
                color = np.ones(3)
                # loop over shapes excluding the ground plane
                for s in range(model.shape_count-1):
                    geo_type = shape_geo_type[s]
                    geo_scale = [float(v) for v in shape_geo_scale[s]]
                    geo_thickness = float(shape_geo_thickness[s])
                    geo_is_solid = bool(shape_geo_is_solid[s])
                    geo_src = shape_geo_src[s]
                    if self.use_unique_colors:
                        color = self._get_new_color()
                    name = f"shape_{s}"

                    # shape transform in body frame
                    body = int(shape_body[s])
                    if body >= 0 and body < len(self.body_names):
                        body = self.body_names[body]
                    else:
                        body = None

                    # shape transform in body frame
                    X_bs = wp.transform_expand(shape_transform[s])
                    # check whether we can instance an already created shape with the same geometry
                    geo_hash = hash((int(geo_type), geo_src, *geo_scale, geo_thickness, geo_is_solid))
                    if geo_hash in self.geo_shape:
                        shape = self.geo_shape[geo_hash]
                    else:
                        if (geo_type == warp.sim.GEO_PLANE):
                            if (s == model.shape_count-1 and not model.ground):
                                continue  # hide ground plane

                            # plane mesh
                            width = (geo_scale[0] if geo_scale[0] > 0.0 else 100.0)
                            length = (geo_scale[1] if geo_scale[1] > 0.0 else 100.0)

                            shape = self.render_plane(name, p, q, width, length, color, parent_body=body, is_template=True)

                        elif (geo_type == warp.sim.GEO_SPHERE):

                            shape = self.render_sphere(name, p, q, geo_scale[0], parent_body=body, is_template=True)

                        elif (geo_type == warp.sim.GEO_CAPSULE):

                            shape = self.render_capsule(name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True)

                        elif (geo_type == warp.sim.GEO_CYLINDER):
                            
                            shape = self.render_cylinder(name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True)

                        elif (geo_type == warp.sim.GEO_CONE):
                            
                            shape = self.render_cone(name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True)

                        elif (geo_type == warp.sim.GEO_BOX):
                            
                            shape = self.render_box(name, p, q, geo_scale, parent_body=body, is_template=True)

                        elif (geo_type == warp.sim.GEO_MESH):

                            if not geo_is_solid:
                                faces, vertices = solidify_mesh(geo_src.indices, geo_src.vertices, geo_thickness)
                            else:
                                faces, vertices = geo_src.indices, geo_src.vertices

                            shape = self.render_mesh(name, vertices, faces, pos=p, rot=q, scale=geo_scale, colors=[color], parent_body=body, is_template=True)

                        elif (geo_type == warp.sim.GEO_SDF):
                            continue

                        self.geo_shape[geo_hash] = shape

                    self.add_shape_instance(name, shape, body, X_bs.p, X_bs.q, scale, color)
                    self.instance_count += 1

            if model.ground:
                self.render_ground()

            if hasattr(self, "complete_setup"):
                self.complete_setup()
    
        def _get_new_color(self):
            return tab10_color_map(self.instance_count)

        def render(self, state: warp.sim.State):

            if self.skip_rendering:
                return

            if (self.model.particle_count):

                particle_q = state.particle_q.numpy()

                # render particles
                self.render_points("particles", particle_q, radius=self.model.soft_contact_distance)

                # render tris
                if (self.model.tri_count):
                    self.render_mesh("surface", particle_q, self.model.tri_indices.numpy().flatten())

                # render springs
                if (self.model.spring_count):
                    self.render_line_list("springs", particle_q, self.model.spring_indices.numpy().flatten(), [], 0.05)

            # render muscles
            if (self.model.muscle_count):
                
                body_q = state.body_q.numpy()

                muscle_start = self.model.muscle_start.numpy()
                muscle_links = self.model.muscle_bodies.numpy()
                muscle_points = self.model.muscle_points.numpy()
                muscle_activation = self.model.muscle_activation.numpy()

                # for s in self.skeletons:
                    
                #     # for mesh, link in s.mesh_map.items():
                        
                #     #     if link != -1:
                #     #         X_sc = wp.transform_expand(self.state.body_X_sc[link].tolist())

                #     #         #self.renderer.add_mesh(mesh, "../assets/snu/OBJ/" + mesh + ".usd", X_sc, 1.0, self.render_time)
                #     #         self.renderer.add_mesh(mesh, "../assets/snu/OBJ/" + mesh + ".usd", X_sc, 1.0, self.render_time)

                for m in range(self.model.muscle_count):

                    start = int(muscle_start[m])
                    end = int(muscle_start[m + 1])

                    points = []

                    for w in range(start, end):
                        
                        link = muscle_links[w]
                        point = muscle_points[w]

                        X_sc = wp.transform_expand(body_q[link][0])

                        points.append(wp.transform_point(X_sc, point).tolist())
                    
                    self.render_line_strip(name=f"muscle_{m}", vertices=points, radius=0.0075, color=(muscle_activation[m], 0.2, 0.5))
           
            # update bodies
            if (self.model.body_count): 
                self.update_body_transforms(state.body_q)

    return SimRenderer_t


class SimRendererUsd2(AbstractSimRenderer(wp.render.UsdRenderer)):

    def register_body(self, body_name):
        from pxr import UsdGeom
        xform = UsdGeom.Xform.Define(self.stage, self.root.GetPath().AppendChild(body_name))
        wp.render.render_usd._usd_add_xform(xform)
    
    def __init__2(self, model: warp.sim.Model, path, scaling=1.0, fps=60, upaxis="y"):

        from pxr import UsdGeom

        # create USD stage
        super().__init__(path, scaling=scaling, fps=fps, upaxis=upaxis)

        self.model = model
        self.body_names = []

        # create rigid body root node
        for b in range(model.body_count):
            body_name = f"body_{b}_{self.model.body_name[b].replace(' ', '_')}"
            self.body_names.append(body_name)
            xform = UsdGeom.Xform.Define(self.stage, self.root.GetPath().AppendChild(body_name))
            wp.render._usd_add_xform(xform)

        # create rigid shape children
        if (self.model.shape_count):
            shape_body = model.shape_body.numpy()
            shape_geo_src = model.shape_geo_src
            shape_geo_type = model.shape_geo.type.numpy()
            shape_geo_scale = model.shape_geo.scale.numpy()
            shape_transform = model.shape_transform.numpy()

            for s in range(model.shape_count):
            
                parent_path = self.root.GetPath()
                body = shape_body[s]
                if body >= 0:
                    parent_path = parent_path.AppendChild(self.body_names[body.item()])

                geo_type = shape_geo_type[s]
                geo_scale = shape_geo_scale[s]
                geo_src = shape_geo_src[s]

                # shape transform in body frame
                X_bs = warp.transform_expand(shape_transform[s])

                if (geo_type == warp.sim.GEO_PLANE):
                    if (s == model.shape_count-1 and not model.ground):
                        continue  # hide ground plane

                    # plane mesh
                    width = (geo_scale[0] if geo_scale[0] > 0.0 else 100.0)
                    length = (geo_scale[1] if geo_scale[1] > 0.0 else 100.0)

                    mesh = UsdGeom.Mesh.Define(self.stage, parent_path.AppendChild("plane_" + str(s)))
                    mesh.CreateDoubleSidedAttr().Set(True)

                    points = ((-width, 0.0, -length), (width, 0.0, -length), (width, 0.0, length), (-width, 0.0, length))
                    normals = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
                    counts = (4, )
                    indices = [0, 1, 2, 3]

                    mesh.GetPointsAttr().Set(np.array(points))
                    mesh.GetNormalsAttr().Set(normals)
                    mesh.GetFaceVertexCountsAttr().Set(counts)
                    mesh.GetFaceVertexIndicesAttr().Set(indices)

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bs.p, X_bs.q, (1.0, 1.0, 1.0), 0.0)

                elif (geo_type == warp.sim.GEO_SPHERE):

                    mesh = UsdGeom.Sphere.Define(self.stage, parent_path.AppendChild("sphere_" + str(s)))
                    mesh.GetRadiusAttr().Set(float(geo_scale[0]))

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bs.p, X_bs.q, (1.0, 1.0, 1.0), 0.0)

                elif (geo_type == warp.sim.GEO_CAPSULE):
                    mesh = UsdGeom.Capsule.Define(self.stage, parent_path.AppendChild("capsule_" + str(s)))
                    mesh.GetRadiusAttr().Set(float(geo_scale[0]))
                    mesh.GetHeightAttr().Set(float(geo_scale[1] * 2.0))

                    # geometry transform w.r.t shape, convert USD geometry to physics engine convention
                    X_sg = warp.transform((0.0, 0.0, 0.0), warp.utils.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5))
                    X_bg = warp.utils.transform_multiply(X_bs, X_sg)

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bg.p, X_bg.q, (1.0, 1.0, 1.0), 0.0)

                elif (geo_type == warp.sim.GEO_CYLINDER):
                    mesh = UsdGeom.Cylinder.Define(self.stage, parent_path.AppendChild("cylinder_" + str(s)))
                    mesh.GetRadiusAttr().Set(float(geo_scale[0]))
                    mesh.GetHeightAttr().Set(float(geo_scale[1] * 2.0))

                    # geometry transform w.r.t shape, convert USD geometry to physics engine convention
                    X_sg = warp.transform((0.0, 0.0, 0.0), warp.utils.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5))
                    X_bg = warp.utils.transform_multiply(X_bs, X_sg)

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bg.p, X_bg.q, (1.0, 1.0, 1.0), 0.0)

                elif (geo_type == warp.sim.GEO_CONE):
                    mesh = UsdGeom.Cone.Define(self.stage, parent_path.AppendChild("cone_" + str(s)))
                    mesh.GetRadiusAttr().Set(float(geo_scale[0]))
                    mesh.GetHeightAttr().Set(float(geo_scale[1] * 2.0))

                    # geometry transform w.r.t shape, convert USD geometry to physics engine convention
                    X_sg = warp.transform((0.0, 0.0, 0.0), warp.utils.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5))
                    X_bg = warp.utils.transform_multiply(X_bs, X_sg)

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bg.p, X_bg.q, (1.0, 1.0, 1.0), 0.0)

                elif (geo_type == warp.sim.GEO_BOX):
                    mesh = UsdGeom.Cube.Define(self.stage, parent_path.AppendChild("box_" + str(s)))
                    #mesh.GetSizeAttr().Set((geo_scale[0], geo_scale[1], geo_scale[2]))

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bs.p, X_bs.q, (geo_scale[0], geo_scale[1], geo_scale[2]), 0.0)

                elif (geo_type == warp.sim.GEO_MESH):

                    mesh = UsdGeom.Mesh.Define(self.stage, parent_path.AppendChild("mesh_" + str(s)))
                    mesh.GetPointsAttr().Set(np.array(geo_src.vertices))
                    mesh.GetFaceVertexIndicesAttr().Set(np.array(geo_src.indices))
                    mesh.GetFaceVertexCountsAttr().Set([3] * int(len(geo_src.indices) / 3))

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bs.p, X_bs.q, (geo_scale[0], geo_scale[1], geo_scale[2]), 0.0)

                elif (geo_type == warp.sim.GEO_SDF):
                    pass

SimRendererUsd = AbstractSimRenderer(wp.render.UsdRenderer)
SimRenderer = SimRendererUsd
SimRendererTiny = AbstractSimRenderer(wp.render.TinyRenderer)
