# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import os

import warp as wp
# import warp.sim
from .utils import tab10_color_map

from collections import defaultdict
from typing import Union

import numpy as np

import glfw
from OpenGL.GL import *
import OpenGL.GL as gl
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import glm
import pycuda
import pycuda.gl
import imgui
from imgui.integrations.glfw import GlfwRenderer

wp.set_module_options({"enable_backward": False})

shape_vertex_shader = '''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

// column vectors of the instance transform matrix
layout (location = 3) in vec4 aInstanceTransform0;
layout (location = 4) in vec4 aInstanceTransform1;
layout (location = 5) in vec4 aInstanceTransform2;
layout (location = 6) in vec4 aInstanceTransform3;

// colors to use for the checkerboard pattern
layout (location = 7) in vec3 aObjectColor1;
layout (location = 8) in vec3 aObjectColor2;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoord;
out vec3 ObjectColor1;
out vec3 ObjectColor2;

void main()
{
    mat4 transform = model * mat4(aInstanceTransform0, aInstanceTransform1, aInstanceTransform2, aInstanceTransform3);
    vec4 worldPos = transform * vec4(aPos, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = vec3(worldPos);
    Normal = mat3(transpose(inverse(transform))) * aNormal;
    TexCoord = aTexCoord;
    ObjectColor1 = aObjectColor1;
    ObjectColor2 = aObjectColor2;
}
'''

shape_fragment_shader = '''
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord;
in vec3 ObjectColor1;
in vec3 ObjectColor2;

uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 sunDirection;

void main()
{
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    vec3 norm = normalize(Normal);

    float diff = max(dot(norm, sunDirection), 0.0);
    vec3 diffuse = diff * lightColor;
    
    vec3 lightDir2 = normalize(vec3(1.0, 0.3, -0.3));
    diff = max(dot(norm, lightDir2), 0.0);
    diffuse += diff * lightColor * 0.3;

    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 reflectDir = reflect(-sunDirection, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    reflectDir = reflect(-lightDir2, norm);
    spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    specular += specularStrength * spec * lightColor * 0.3;
    
    // checkerboard pattern
    float u = TexCoord.x;
    float v = TexCoord.y;
    // blend the checkerboard pattern dependent on the gradient of the texture coordinates
    // to void Moire patterns
    vec2 grad = abs(dFdx(TexCoord)) + abs(dFdy(TexCoord));
    float blendRange = 1.5;
    float blendFactor = max(grad.x, grad.y) * blendRange;
    float scale = 2.0;
    float checker = mod(floor(u * scale) + floor(v * scale), 2.0);
    checker = mix(checker, 0.5, smoothstep(0.0, 1.0, blendFactor));
    vec3 checkerColor = mix(ObjectColor1, ObjectColor2, checker);

    vec3 result = (ambient + diffuse + specular) * checkerColor;
    FragColor = vec4(result, 1.0);
}
'''

grid_vertex_shader = '''
#version 330 core

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

in vec3 position;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
}
'''

# Fragment shader source code
grid_fragment_shader = '''
#version 330 core

out vec4 outColor;

void main() {
    outColor = vec4(0.5, 0.5, 0.5, 1.0);
}
'''

sky_vertex_shader = '''
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;
uniform vec3 viewPos;

out vec3 FragPos;
out vec2 TexCoord;

void main()
{
    vec4 worldPos = vec4(aPos + viewPos, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = vec3(worldPos);
    TexCoord = aTexCoord;
}
'''

sky_fragment_shader = '''
#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCoord;

uniform vec3 color1;
uniform vec3 color2;

uniform vec3 sunDirection;

void main()
{
    float y = tanh(FragPos.y*0.01)*0.5+0.5;
    float height = sqrt(1.0-y);
    
    float s = pow(0.5, 1.0 / 10.0);
    s = 1.0 - clamp(s, 0.75, 1.0);
    
    vec3 haze = mix(vec3(1.0), color2 * 1.3, s);
    vec3 sky = mix(color1, haze, height / 1.3);

    float diff = max(dot(sunDirection, normalize(FragPos)), 0.0);
    vec3 sun = pow(diff, 32) * vec3(1.0, 0.8, 0.6) * 0.5;

	FragColor = vec4(sky + sun, 1.0);
}
'''

frame_vertex_shader = '''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
'''

frame_fragment_shader = '''
#version 330 core
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D textureSampler;

void main() {
    FragColor = texture(textureSampler, TexCoord);
}
'''


@wp.kernel
def move_instances(
    positions: wp.array(dtype=wp.vec3),
    scalings: wp.array(dtype=wp.vec3),
    time: float,
    # outputs
    transforms: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    angle = (36.0 * float(tid))*wp.pi/180.0 + time
    axis = wp.vec3(0., 1., 0.)
    rot = wp.quat_to_matrix(wp.quat_from_axis_angle(axis, angle))
    offset = wp.vec3(0.0, 0.0, 5.0 * np.sin(float(tid)*wp.pi/4. + time))
    position = positions[tid] + offset
    scaling = scalings[tid]
    scale = wp.mat33(scaling[0], 0.0, 0.0, 0.0, scaling[1], 0.0, 0.0, 0.0, scaling[2])
    scaled_rot = scale * rot
    transforms[tid] = wp.transpose(wp.mat44(
        scaled_rot[0,0], scaled_rot[0,1], scaled_rot[0,2], position[0],
        scaled_rot[1,0], scaled_rot[1,1], scaled_rot[1,2], position[1],
        scaled_rot[2,0], scaled_rot[2,1], scaled_rot[2,2], position[2],
        0.0, 0.0, 0.0, 1.0))


@wp.kernel
def update_vbo_transforms(
    instance_id: wp.array(dtype=int),
    instance_body: wp.array(dtype=int),
    instance_transforms: wp.array(dtype=wp.transform),
    instance_scalings: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44)):

    tid = wp.tid()
    i = instance_id[tid]
    X_ws = instance_transforms[i]
    if instance_body:
        body = instance_body[i]
        if body >= 0:
            if body_q:
                X_ws = body_q[body] * X_ws
    p = wp.transform_get_translation(X_ws)
    q = wp.transform_get_rotation(X_ws)
    s = instance_scalings[i]
    rot = wp.quat_to_matrix(q)
    # transposed definition
    vbo_transforms[tid] = wp.mat44(
        rot[0,0]*s[0], rot[1,0]*s[1], rot[2,0]*s[2], 0.0,
        rot[0,1]*s[0], rot[1,1]*s[1], rot[2,1]*s[2], 0.0,
        rot[0,2]*s[0], rot[1,2]*s[1], rot[2,2]*s[2], 0.0,
        p[0], p[1], p[2], 1.0)


@wp.kernel
def update_points_positions(
    instance_positions: wp.array(dtype=wp.vec3),
    instance_scalings: wp.array(dtype=wp.vec3),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44)):

    tid = wp.tid()
    p = instance_positions[tid]
    s = wp.vec3(1.0)
    if instance_scalings:
        s = instance_scalings[tid]
    # transposed definition
    vbo_transforms[tid] = wp.mat44(
        s[0],  0.0,  0.0, 0.0,
        0.0,  s[1],  0.0, 0.0,
        0.0,   0.0, s[2], 0.0,
        p[0], p[1], p[2], 1.0)


@wp.kernel
def update_line_transforms(
    instance_id: wp.array(dtype=int),
    lines: wp.array(dtype=wp.vec3, ndim=2),
    scaling: float,
    # outputs
    vbo_positions: wp.array(dtype=wp.vec4),
    vbo_orientations: wp.array(dtype=wp.quat),
    vbo_scalings: wp.array(dtype=wp.vec4)):

    tid = wp.tid()
    p0 = lines[tid, 0]
    p1 = lines[tid, 1]
    p = (p0 + p1) * (0.5 * scaling)
    d = p1 - p0
    s = wp.length(d)
    axis = wp.normalize(d)
    y_up = wp.vec3(0.0, 1.0, 0.0)
    angle = wp.acos(wp.dot(axis, y_up))
    axis = wp.normalize(wp.cross(axis, y_up))
    q = wp.quat_from_axis_angle(axis, -angle)
    i = instance_id[tid]
    vbo_positions[i] = wp.vec4(p[0], p[1], p[2], 0.0)
    vbo_orientations[i] = q
    vbo_scalings[i] = wp.vec4(1.0, s, 1.0, 1.0)


@wp.kernel
def compute_gfx_vertices(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    # outputs
    gfx_vertices: wp.array(dtype=float, ndim=2)):

    tid = wp.tid()
    v0 = vertices[indices[tid, 0]]
    v1 = vertices[indices[tid, 1]]
    v2 = vertices[indices[tid, 2]]
    i = tid * 3; j = i + 1; k = i + 2
    gfx_vertices[i,0] = v0[0]; gfx_vertices[i,1] = v0[1]; gfx_vertices[i,2] = v0[2]
    gfx_vertices[j,0] = v1[0]; gfx_vertices[j,1] = v1[1]; gfx_vertices[j,2] = v1[2]
    gfx_vertices[k,0] = v2[0]; gfx_vertices[k,1] = v2[1]; gfx_vertices[k,2] = v2[2]
    n = wp.normalize(wp.cross(v1-v0, v2-v0))
    gfx_vertices[i,3] = n[0]; gfx_vertices[i,4] = n[1]; gfx_vertices[i,5] = n[2]
    gfx_vertices[j,3] = n[0]; gfx_vertices[j,4] = n[1]; gfx_vertices[j,5] = n[2]
    gfx_vertices[k,3] = n[0]; gfx_vertices[k,4] = n[1]; gfx_vertices[k,5] = n[2]


@wp.kernel
def compute_average_normals(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3),
    # outputs
    normals: wp.array(dtype=wp.vec3),
    faces_per_vertex: wp.array(dtype=int)):

    tid = wp.tid()
    i = indices[tid, 0]; j = indices[tid, 1]; k = indices[tid, 2]
    v0 = vertices[i]
    v1 = vertices[j]
    v2 = vertices[k]
    n = wp.normalize(wp.cross(v1-v0, v2-v0))
    wp.atomic_add(normals, i, n); wp.atomic_add(faces_per_vertex, i, 1)
    wp.atomic_add(normals, j, n); wp.atomic_add(faces_per_vertex, j, 1)
    wp.atomic_add(normals, k, n); wp.atomic_add(faces_per_vertex, k, 1)


@wp.kernel
def assemble_gfx_vertices(
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    normals: wp.array(dtype=wp.vec3),
    faces_per_vertex: wp.array(dtype=int),
    # outputs
    gfx_vertices: wp.array(dtype=float, ndim=2)):

    tid = wp.tid()    
    v = vertices[tid]
    n = normals[tid] / float(faces_per_vertex[tid])
    gfx_vertices[tid,0] = v[0]; gfx_vertices[tid,1] = v[1]; gfx_vertices[tid,2] = v[2]
    gfx_vertices[tid,3] = n[0]; gfx_vertices[tid,4] = n[1]; gfx_vertices[tid,5] = n[2]


@wp.kernel
def copy_frame(
    input_img: wp.array(dtype=float),
    width: int,
    height: int,
    # outputs
    output_img: wp.array(dtype=float, ndim=3)):

    w, v = wp.tid()
    pixel = v*width + w
    pixel *= 3
    r = input_img[pixel+0]; g = input_img[pixel+1]; b = input_img[pixel+2]
    # flip vertically (OpenGL coordinates start at bottom)
    v = height - v - 1
    output_img[v, w, 0] = r
    output_img[v, w, 1] = g
    output_img[v, w, 2] = b


@wp.kernel
def copy_frame_tiles(
    input_img: wp.array(dtype=float),
    screen_width: int,
    tile_ncols: int,
    tile_width: int,
    tile_height: int,
    # outputs
    output_img: wp.array(dtype=float, ndim=4)):

    tile, w, v = wp.tid()
    row = tile_ncols - tile // tile_ncols - 1
    col = tile % tile_ncols
    pixel = v*screen_width + w + row * tile_height * screen_width + col * tile_width
    pixel *= 3
    r = input_img[pixel+0]; g = input_img[pixel+1]; b = input_img[pixel+2]
    # flip vertically (OpenGL coordinates start at bottom)
    v = tile_height - v - 1
    output_img[tile, v, w, 0] = r
    output_img[tile, v, w, 1] = g
    output_img[tile, v, w, 2] = b


def check_gl_error():
    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        print(f"OpenGL error: {error}")


class ShapeInstancer:
    """
    Handles instanced rendering for a mesh.
    Note the vertices must be in the 8-dimensional format:
        [3D point, 3D normal, UV texture coordinates]
    """
    def __init__(self, shape_shader, device):
        self.shape_shader = shape_shader
        self.device = device
        self.face_count = 0
        self.vao = None
        self.instance_transform_gl_buffer = None
        self.instance_color1_buffer = None
        self.instance_color2_buffer = None
        self.color1 = (1., 1., 1.)
        self.color2 = (0., 0., 0.)
        self.num_instances = 0
        self.transforms = None
        self.scalings = None
        self.instance_cuda_buffer = None

    def __del__(self):
        if self.instance_cuda_buffer is not None:
            self.instance_cuda_buffer.unregister()
        if self.instance_transform_gl_buffer is not None:
            glDeleteBuffers(1, [self.instance_transform_gl_buffer])
            glDeleteBuffers(1, [self.instance_color1_buffer])
            glDeleteBuffers(1, [self.instance_color2_buffer])
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            glDeleteBuffers(1, [self.vbo])
            glDeleteBuffers(1, [self.ebo])

    def register_shape(self, vertices, indices, color1, color2):
        if color1 is not None:
            color1 = color1
        elif color2 is None:
            color2 = np.clip(np.array(color1) + 0.25, 0.0, 1.0)
        self.color1 = color1
        self.color2 = color2

        glUseProgram(self.shape_shader)

        # Create VAO, VBO, and EBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.flatten(), GL_STATIC_DRAW)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # normals
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        glEnableVertexAttribArray(1)
        # uv coordinates
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

        self.face_count = len(indices)

    def allocate_instances(self, positions, rotations=None, colors1=None, colors2=None, scalings=None):
        glBindVertexArray(self.vao)

        self.num_instances = len(positions)

        # Create instance buffer and bind it as an instanced array
        if self.instance_transform_gl_buffer is None:
            self.instance_transform_gl_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_transform_gl_buffer)

        transforms = np.tile(np.diag(np.ones(4, dtype=np.float32)), (self.num_instances, 1, 1))
        glBufferData(GL_ARRAY_BUFFER, transforms.nbytes, transforms, GL_DYNAMIC_DRAW)

        # Create CUDA buffer for instance transforms
        self.instance_cuda_buffer = pycuda.gl.RegisteredBuffer(int(self.instance_transform_gl_buffer))

        # Upload instance matrices to GPU
        # offset = 0
        # for matrix in transforms:
        #     glBufferSubData(GL_ARRAY_BUFFER, offset, matrix.nbytes, matrix)
        #     offset += matrix.nbytes

        if colors1 is None:
            colors1 = np.tile(self.color1, (self.num_instances, 1))
        if colors2 is None:
            colors2 = np.tile(self.color2, (self.num_instances, 1))
        colors1 = np.array(colors1, dtype=np.float32)
        colors2 = np.array(colors2, dtype=np.float32)

        # colors1 = np.array([instance[5] for instance in self._instances.values()], dtype=np.float32)
        # colors2 = np.array([instance[6] for instance in self._instances.values()], dtype=np.float32)

        # create buffer for checkerboard colors
        if self.instance_color1_buffer is None:
           self.instance_color1_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_color1_buffer)
        glBufferData(GL_ARRAY_BUFFER, colors1.nbytes, colors1.flatten(), GL_STATIC_DRAW)
        
        if self.instance_color2_buffer is None:
            self.instance_color2_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_color2_buffer)
        glBufferData(GL_ARRAY_BUFFER, colors2.nbytes, colors2.flatten(), GL_STATIC_DRAW)

        # Set up instance attribute pointers
        matrix_size = transforms[0].nbytes
        
        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.instance_transform_gl_buffer)

        # we can only send vec4s to the shader, so we need to split the instance transforms matrix into its column vectors
        for i in range(4):
            glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, matrix_size, ctypes.c_void_p(i * matrix_size // 4))
            glEnableVertexAttribArray(3 + i)
            glVertexAttribDivisor(3 + i, 1)

        glBindBuffer(GL_ARRAY_BUFFER, self.instance_color1_buffer)
        glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, colors1[0].nbytes, ctypes.c_void_p(0))
        glEnableVertexAttribArray(7)
        glVertexAttribDivisor(7, 1)
    
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_color2_buffer)
        glVertexAttribPointer(8, 3, GL_FLOAT, GL_FALSE, colors2[0].nbytes, ctypes.c_void_p(0))
        glEnableVertexAttribArray(8)
        glVertexAttribDivisor(8, 1)
        
        # trigger update to the instance transforms
        self._update_shape_instances = True

        self.instance_ids = wp.array(np.arange(self.num_instances), dtype=wp.int32, device=self.device)
        if rotations is None:
            self.instance_transforms = wp.array([(*pos, 0., 0., 0., 1.) for pos in positions], dtype=wp.transform, device=self.device)
        else:
            self.instance_transforms = wp.array([
                (*pos, *rot) for pos, rot in zip(positions, rotations)
            ], dtype=wp.transform, device=self.device)

        if scalings is None:
            self.instance_scalings = wp.array(np.tile((1., 1., 1.), (self.num_instances, 1)), dtype=wp.vec3, device=self.device)
        else:
            self.instance_scalings = wp.array(scalings, dtype=wp.vec3, device=self.device)

        glBindVertexArray(0)

    def update_instances(self, transforms: wp.array=None, scalings: wp.array=None, colors1=None, colors2=None):
        if transforms is not None:
            if transforms.device.is_cuda:
                wp_transforms = transforms
            else:
                wp_transforms = transforms.to(self.device)
            self.transforms = wp_transforms
        if scalings is not None:
            if transforms.device.is_cuda:
                wp_scalings = scalings
            else:
                wp_scalings = scalings.to(self.device)
            self.scalings = wp_scalings

        if transforms is not None or scalings is not None:
            glBindVertexArray(self.vao)
            mapped_buffer = self.instance_cuda_buffer.map()
            ptr, _ = mapped_buffer.device_ptr_and_size()
            vbo_transforms = wp.array(dtype=wp.mat44, shape=(self.num_instances,), device=self.device, ptr=ptr, owner=False)

            wp.launch(
                update_vbo_transforms,
                dim=self.num_instances,
                inputs=[
                    self.instance_ids,
                    None,
                    self.instance_transforms,
                    self.instance_scalings,
                    None,
                ],
                outputs=[
                    vbo_transforms,
                ],
                device=self.device)
            
            mapped_buffer.unmap()

    def render(self):
        glUseProgram(self.shape_shader)

        glBindVertexArray(self.vao)
        glDrawElementsInstanced(GL_TRIANGLES, self.face_count, GL_UNSIGNED_INT, None, self.num_instances)
        glBindVertexArray(0)

    # scope exposes VBO transforms to be set directly by a warp kernel
    def __enter__(self):
        glBindVertexArray(self.vao)
        self._mapped_buffer = self.instance_cuda_buffer.map()
        ptr, _ = self._mapped_buffer.device_ptr_and_size()
        self.vbo_transforms = wp.array(dtype=wp.mat44, shape=(self.num_instances,), device=self.device, ptr=ptr, owner=False)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._mapped_buffer.unmap()


class TinyRenderer:
    # number of segments to use for rendering spheres, capsules, cones and cylinders
    default_num_segments = 32

    # number of horizontal and vertical pixels to use for checkerboard texture
    default_texture_size = 256

    def __init__(
        self,
        title="Warp sim",
        scaling=1.0,
        fps=60,
        upaxis="y",
        screen_width=1024,
        screen_height=768,
        near_plane=0.01,
        far_plane=1000.0,
        camera_fov=45.0,
        background_color=(0.53, 0.8, 0.92),
        draw_grid=True,
        draw_sky=True,
        draw_axis=True,
        axis_scale=1.0,
        use_vsync=True,
        headless=False,
    ):
        
        self.scaling = scaling
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.camera_fov = camera_fov
        self.background_color = background_color
        self.draw_grid = draw_grid
        self.draw_sky = draw_sky
        self.draw_axis = draw_axis
        
        self.screen_width = screen_width
        self.screen_height = screen_height

        self._device = wp.get_cuda_device()

        if not glfw.init():
            raise Exception("GLFW initialization failed!")
        
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # increase depth buffer precision to preven z fighting
        glfw.window_hint(glfw.DEPTH_BITS, 32)
        if headless:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self.window = glfw.create_window(screen_width, screen_height, title, None, None)

        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed!")
        
        self._camera_pos = glm.vec3(0.0, 2.0, 10.0)
        self._camera_front = glm.vec3(0.0, 0.0, -1.0)
        self._camera_up = glm.vec3(0.0, 1.0, 0.0)
        self._camera_speed = 0.04
        self._camera_axis = "xyz".index(upaxis.lower())
        self._yaw, self._pitch = -90.0, 0.0
        self._last_x, self._last_y = 800 // 2, 600 // 2
        self._first_mouse = True
        self._left_mouse_pressed = False
        self._keys_pressed = defaultdict(bool)

        self.time = 0.0
        self.clock_time = glfw.get_time()
        self.paused = False
        self._frame_speed = 0.0
        self.skip_rendering = False
        self._skip_frame_counter = 0
        self._fps_update = None
        self._fps_render = None
        self._fps_alpha = 0.1  # low pass filter update

        self._body_name = {}
        self._shapes = []
        self._shape_geo_hash = {}
        self._shape_gl_buffers = {}
        self._shape_instances = defaultdict(list)
        self._instances = {}
        self._instance_shape = {}
        self._instance_gl_buffers = {}
        self._instance_transform_gl_buffer = None
        self._instance_transform_cuda_buffer = None
        self._instance_color1_buffer = None
        self._instance_color2_buffer = None
        self._instance_count = 0
        self._wp_instance_ids = None
        self._instance_ids = None
        self._inverse_instance_ids = None
        self._wp_instance_transforms = None
        self._wp_instance_scalings = None
        self._wp_instance_bodies = None
        self._update_shape_instances = False
        self._add_shape_instances = False

        # additional shape instancer used for points and line rendering
        self._shape_instancers = {}

        self._tile_instances = None
        self._tile_ncols = 0
        self._tile_nrows = 0
        self._tile_width = 0
        self._tile_height = 0

        self._frame_texture = None
        self._frame_fbo = None
        self._frame_pbo = None

        glfw.make_context_current(self.window)
        if not use_vsync:
            glfw.swap_interval(0)

        # Initialize Dear ImGui and the OpenGL renderer
        imgui.create_context()
        self.imgui_io = imgui.get_io()
        self.imgui_renderer = GlfwRenderer(self.window)
        
        # glfw.set_window_pos(self.window, 400, 200)
        glfw.set_window_size_callback(self.window, self._window_resize_callback)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        
        glClearColor(*self.background_color, 1)
        glEnable(GL_DEPTH_TEST)

        self._shape_shader = compileProgram(
            compileShader(shape_vertex_shader, GL_VERTEX_SHADER),
            compileShader(shape_fragment_shader, GL_FRAGMENT_SHADER)
        )
        self._grid_shader = compileProgram(
            compileShader(grid_vertex_shader, GL_VERTEX_SHADER),
            compileShader(grid_fragment_shader, GL_FRAGMENT_SHADER)
        )

        glUseProgram(self._shape_shader)

        self._loc_shape_model = glGetUniformLocation(self._shape_shader, "model")
        self._loc_shape_view = glGetUniformLocation(self._shape_shader, "view")
        self._loc_shape_projection = glGetUniformLocation(self._shape_shader, "projection")
        self._loc_shape_view_pos = glGetUniformLocation(self._shape_shader, "viewPos")
        glUniform3f(glGetUniformLocation(self._shape_shader, "lightColor"), 1, 1, 1)
        glUniform3f(self._loc_shape_view_pos, 0, 0, 10)

        self._sun_direction = np.array((-0.2, 0.8, 0.3))
        self._sun_direction /= np.linalg.norm(self._sun_direction)
        glUniform3f(glGetUniformLocation(self._shape_shader, "sunDirection"), *self._sun_direction)

        width, height = glfw.get_window_size(self.window)
        self._projection_matrix = glm.perspective(np.deg2rad(45), width / height, self.near_plane, self.far_plane)
        # glUniformMatrix4fv(self._loc_shape_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))

        self._view_matrix = glm.lookAt(self._camera_pos, self._camera_pos + self._camera_front, self._camera_up)
        # glUniformMatrix4fv(self._loc_shape_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        
        if self._camera_axis == 0:
            self._model_matrix = glm.mat4(
                0, 0, self.scaling, 0,
                self.scaling, 0, 0, 0,
                0, self.scaling, 0, 0,
                0, 0, 0, 1)
        elif self._camera_axis == 2:
            self._model_matrix = glm.mat4(
                self.scaling, 0, 0, 0,
                0, 0, self.scaling, 0,
                0, self.scaling, 0, 0,
                0, 0, 0, 1)
        else:
            self._model_matrix = glm.mat4(
                self.scaling, 0, 0, 0,
                0, self.scaling, 0, 0,
                0, 0, self.scaling, 0,
                0, 0, 0, 1)

        glUniformMatrix4fv(self._loc_shape_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))


        glUseProgram(self._grid_shader)

        # create grid data
        limit = 10.0
        ticks = np.linspace(-limit, limit, 21)
        grid_vertices = []
        for i in ticks:
            if self._camera_axis == 0:
                grid_vertices.extend([0, -limit, i, 0, limit, i])
                grid_vertices.extend([0, i, -limit, 0, i, limit])
            elif self._camera_axis == 1:
                grid_vertices.extend([-limit, 0, i, limit, 0, i])
                grid_vertices.extend([i, 0, -limit, i, 0, limit])
            elif self._camera_axis == 2:
                grid_vertices.extend([-limit, i, 0, limit, i, 0])
                grid_vertices.extend([i, -limit, 0, i, limit, 0])
        grid_vertices = np.array(grid_vertices, dtype=np.float32)
        self._grid_vertex_count = len(grid_vertices) // 3

        # glUseProgram(self._grid_shader)
        self._grid_vao = glGenVertexArrays(1)
        glBindVertexArray(self._grid_vao)

        self._grid_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._grid_vbo)
        glBufferData(GL_ARRAY_BUFFER, grid_vertices.nbytes, grid_vertices, GL_STATIC_DRAW)

        self._loc_grid_view = glGetUniformLocation(self._grid_shader, "view")
        self._loc_grid_model = glGetUniformLocation(self._grid_shader, "model")
        self._loc_grid_projection = glGetUniformLocation(self._grid_shader, "projection")
        
        self._loc_grid_pos_attribute = glGetAttribLocation(self._grid_shader, "position")
        glVertexAttribPointer(self._loc_grid_pos_attribute, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(self._loc_grid_pos_attribute)
        
        glUniformMatrix4fv(self._loc_grid_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))
        glUniformMatrix4fv(self._loc_grid_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        glUniformMatrix4fv(self._loc_grid_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))

        # create sky data
        self._sky_shader = compileProgram(
            compileShader(sky_vertex_shader, GL_VERTEX_SHADER),
            compileShader(sky_fragment_shader, GL_FRAGMENT_SHADER)
        )
        
        glUseProgram(self._sky_shader)
        self._loc_sky_view = glGetUniformLocation(self._sky_shader, "view")
        self._loc_sky_model = glGetUniformLocation(self._sky_shader, "model")
        self._loc_sky_projection = glGetUniformLocation(self._sky_shader, "projection")
        glUniformMatrix4fv(self._loc_sky_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))
        glUniformMatrix4fv(self._loc_sky_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        glUniformMatrix4fv(self._loc_sky_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))

        self._loc_sky_color1 = glGetUniformLocation(self._sky_shader, "color1")
        self._loc_sky_color2 = glGetUniformLocation(self._sky_shader, "color2")
        glUniform3f(self._loc_sky_color1, *background_color)
        glUniform3f(self._loc_sky_color2, *np.clip(np.array(background_color)+0.5, 0.0, 1.0))
        glUniform3f(self._loc_sky_color2, 0.8, 0.4, 0.05)
        self._loc_sky_view_pos = glGetUniformLocation(self._sky_shader, "viewPos")
        glUniform3f(glGetUniformLocation(self._sky_shader, "sunDirection"), *self._sun_direction)

        # Create VAO, VBO, and EBO
        self._sky_vao = glGenVertexArrays(1)
        glBindVertexArray(self._sky_vao)

        vertices, indices = self._create_sphere_mesh(self.far_plane * 0.9, 32, 32)
        self._sky_tri_count = len(indices)

        self._sky_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._sky_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.flatten(), GL_STATIC_DRAW)

        self._sky_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._sky_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # normals
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        glEnableVertexAttribArray(1)
        # uv coordinates
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

        import pycuda.gl.autoinit




        # # Clean up
        # glDeleteVertexArrays(1, [vao])
        # glDeleteBuffers(1, [vbo])
        # glDeleteBuffers(1, [ebo])
        # glDeleteBuffers(1, [instance_buffer])
        # glfw.terminate()

        self._last_time = glfw.get_time()
        self._last_begin_frame_time = self._last_time
        self._last_end_frame_time = self._last_time

        glUseProgram(self._shape_shader)
        
        # create arrow shapes for the coordinate system axes
        vertices, indices = self._create_arrow_mesh(base_radius=0.02*axis_scale, base_height=0.85*axis_scale, cap_height=0.15*axis_scale)
        shape = self.register_shape(123, vertices, indices)
        ayi = self.add_shape_instance("arrow_y", shape, -1, (0., 0., 0.), (0., 0., 0., 1.), color1=(0., 1., 0.), color2=(0., 1., 0.))
        sqh = np.sqrt(0.5)
        ayx = self.add_shape_instance("arrow_x", shape, -1, (0., 0., 0.), (0.0, 0.0, -sqh, sqh), color1=(1., 0., 0.), color2=(1., 0., 0.))
        ayz = self.add_shape_instance("arrow_z", shape, -1, (0., 0., 0.), (sqh, 0.0, 0.0, sqh), color1=(0., 0., 1.), color2=(0., 0., 1.))
        self._axis_instances = [ayi, ayx, ayz]


        # create frame buffer for rendering to a texture
        self._frame_texture = None
        self._frame_fbo = None
        self._setup_framebuffer()
        
        # self._frame_cuda_buffer = pycuda.gl.RegisteredImage(self._frame_texture, GL_TEXTURE_2D, pycuda.gl.graphics_map_flags.READ_ONLY)

        # set up VBO for the quad that is rendered to the user window with the texture
        self._frame_vertices = np.array([
            # Positions  TexCoords
            -1.0, -1.0,  0.0, 0.0,
            1.0, -1.0,  1.0, 0.0,
            1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ], dtype=np.float32)

        self._frame_indices = np.array([
            0, 1, 2,
            2, 3, 0
        ], dtype=np.uint32)

        self._frame_vao = glGenVertexArrays(1)
        glBindVertexArray(self._frame_vao)

        self._frame_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._frame_vbo)
        glBufferData(GL_ARRAY_BUFFER, self._frame_vertices.nbytes, self._frame_vertices, GL_STATIC_DRAW)

        self._frame_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._frame_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._frame_indices.nbytes, self._frame_indices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * self._frame_vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * self._frame_vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize))
        glEnableVertexAttribArray(1)
        
        self._frame_shader = compileProgram(
            compileShader(frame_vertex_shader, GL_VERTEX_SHADER),
            compileShader(frame_fragment_shader, GL_FRAGMENT_SHADER)
        )
        glUseProgram(self._frame_shader)
        self._frame_loc_texture = glGetUniformLocation(self._frame_shader, "textureSampler")

        # Unbind the VBO and VAO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _setup_framebuffer(self):
        if self._frame_texture is None:
            self._frame_texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self._frame_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.screen_width, self.screen_height, 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Create a framebuffer object (FBO)
        if self._frame_fbo is None:
            self._frame_fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self._frame_fbo)

            # Attach the texture to the FBO as its color attachment
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._frame_texture, 0)
            
            self._frame_depth_renderbuffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self._frame_depth_renderbuffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.screen_width, self.screen_height)

            # Attach the depth renderbuffer to the FBO
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._frame_depth_renderbuffer)

            # Check if the framebuffer is complete
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print("Framebuffer is not complete!")
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                sys.exit(1)

            glBindRenderbuffer(GL_RENDERBUFFER, 0)
        else:
            glBindRenderbuffer(GL_RENDERBUFFER, self._frame_depth_renderbuffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.screen_width, self.screen_height)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)

        # Unbind the FBO (switch back to the default framebuffer)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        if self._frame_pbo is None:
            self._frame_pbo = glGenBuffers(1) # generate 1 buffer reference
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._frame_pbo) # binding to this buffer
        # allocate memory for PBO
        pixels = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.float32)
        glBufferData(GL_PIXEL_PACK_BUFFER, pixels.nbytes, pixels, GL_DYNAMIC_DRAW) # Allocate the buffer
        # bsize = glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER, GL_BUFFER_SIZE) # Check allocated buffer size
        # assert bsize == pixels.nbytes
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0) # Unbind

    def clear(self):
        if self._instance_transform_gl_buffer is not None:
            self._instance_transform_cuda_buffer.unregister()
            glDeleteBuffers(1, [self._instance_transform_gl_buffer])
            glDeleteBuffers(1, [self._instance_color1_buffer])
            glDeleteBuffers(1, [self._instance_color2_buffer])
        for vao, vbo, ebo, _ in self._shape_gl_buffers.values():
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
            glDeleteBuffers(1, [ebo])
        
        self._body_name.clear()
        self._shapes.clear()
        self._shape_geo_hash.clear()
        self._shape_gl_buffers.clear()
        self._shape_instances.clear()
        self._instances.clear()
        self._instance_shape.clear()
        self._instance_gl_buffers.clear()
        self._instance_transform_gl_buffer = None
        self._instance_transform_cuda_buffer = None
        self._instance_color1_buffer = None
        self._instance_color2_buffer = None
        self._wp_instance_ids = None
        self._wp_instance_transforms = None
        self._wp_instance_scalings = None
        self._wp_instance_bodies = None
        self._update_shape_instances = False

    def setup_tiled_rendering(self, instances: list, rescale_window=False, tile_width=128, tile_height=128):
        # skip the first 3 instances for the coordinate system axis arrows
        num_axes = len(self._axis_instances)
        self._tile_instances = [[i+num_axes for i in ids] + self._axis_instances for ids in instances]
        n = len(self._tile_instances)
        # try to fit the tiles into a square
        self._tile_ncols = int(np.ceil(np.sqrt(n)))
        self._tile_nrows = int(np.ceil(n / float(self._tile_ncols)))
        if rescale_window:
            glfw.set_window_size(self.window, tile_width * self._tile_ncols, tile_height * self._tile_nrows)
        self._tile_width = max(32, self.screen_width // self._tile_ncols)
        self._tile_height = max(32, self.screen_height // self._tile_nrows)

    def update_projection_matrix(self):
        if self._tile_ncols is None or self._tile_ncols == 0:
            if self.screen_height == 0:
                return
            aspect_ratio = self.screen_width / self.screen_height
        else:
            self._tile_width = max(32, self.screen_width // self._tile_ncols)
            self._tile_height = max(32, self.screen_height // self._tile_nrows)
            aspect_ratio = self._tile_width / self._tile_height
        
        self._projection_matrix = glm.perspective(glm.radians(self.camera_fov), aspect_ratio, self.near_plane, self.far_plane)

    @property
    def num_tiles(self):
        return len(self._tile_instances)
    
    @property
    def tile_width(self):
        return self._tile_width
    
    @property
    def tile_height(self):
        return self._tile_height

    @property
    def num_shapes(self):
        return len(self._shapes)
    
    @property
    def num_instances(self):
        return self._instance_count
    
    def begin_frame(self, time: float):
        self._last_begin_frame_time = glfw.get_time()
        self.time = time

    def end_frame(self):
        self._last_end_frame_time = glfw.get_time()
        if self._add_shape_instances:
            self.allocate_shape_instances()
        if self._update_shape_instances:
            self.update_shape_instances()
        self.update()
        while self.paused and self.is_running():
            self.update()

    def update(self):
        self.clock_time = glfw.get_time()
        duration = self.clock_time - self._last_time
        self._last_time = self.clock_time
        self._frame_speed = duration * 100.0

        self._skip_frame_counter += 1
        if self._skip_frame_counter > 100:
            self._skip_frame_counter = 0
        if self.skip_rendering:
            if self._skip_frame_counter == 0:
                # ensure we receive key events
                glfw.poll_events()
                self._process_input(self.window)
            return
        
        glfw.poll_events()
        self.imgui_renderer.process_inputs()
        self._process_input(self.window)

        if self._frame_fbo is not None:
            glBindFramebuffer(GL_FRAMEBUFFER, self._frame_fbo)
        
        glClearColor(*self.background_color, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindVertexArray(0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._frame_fbo)

        # cache camera position in case it gets changed during rendering
        cam_pos = self._camera_pos
        self._view_matrix = glm.lookAt(cam_pos, cam_pos + self._camera_front, self._camera_up)

        if self._fps_update is None:
            self._fps_update = 1.0 / (self._last_end_frame_time - self._last_begin_frame_time)
        else:
            update = 1.0 / (self._last_end_frame_time - self._last_begin_frame_time)
            self._fps_update = (1.0 - self._fps_alpha) * self._fps_update + self._fps_alpha * update
        if self._fps_render is None:
            self._fps_render = 1.0 / duration
        else:
            update = 1.0 / duration
            self._fps_render = (1.0 - self._fps_alpha) * self._fps_render + self._fps_alpha * update

        imgui.new_frame()
        imgui.set_next_window_bg_alpha(0.8)
        imgui.set_next_window_position(0, 0)
        # imgui.set_next_window_size(180, 110)
        imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)
        imgui.begin("Stats", True,
                    imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        imgui.text(f"Sim Time: {self.time:.1f}")
        imgui.spacing()
        imgui.text(f"Update FPS: {self._fps_update:.1f}")
        imgui.text(f"Render FPS: {self._fps_render:.1f}")
        imgui.spacing()
        imgui.text(f"Shapes: {len(self._shapes)}")
        imgui.text(f"Instances: {len(self._instances)}")
        if self.paused:
            imgui.spacing()
            imgui.text("Paused (press space to resume)")
        # string_var = "test"
        # float_var = np.pi
        # if imgui.button("OK"):
        #     print(f"String: {string_var}")
        #     print(f"Float: {float_var}")
        # _, string_var = imgui.input_text("A String", string_var, 256)
        # _, float_var = imgui.slider_float("float", float_var, 0.25, 1.5)
        # imgui.show_test_window()
        imgui.end()
        imgui.pop_style_var()
        imgui.render()

        if self._tile_instances is None:
            if self.draw_grid:
                self._draw_grid()

            if self.draw_sky:
                self._draw_sky()
        
        glUseProgram(self._shape_shader)
        glUniformMatrix4fv(self._loc_shape_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        glUniform3f(self._loc_shape_view_pos, *cam_pos)
        glUniformMatrix4fv(self._loc_shape_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        glUniformMatrix4fv(self._loc_shape_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))
        glUniformMatrix4fv(self._loc_shape_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))

        if self._tile_instances is None:
            self._render_scene()
        else:
            self._render_scene_tiled()
        
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        if self._frame_fbo is not None:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glUseProgram(self._frame_shader)
            glViewport(0, 0, self.screen_width, self.screen_height)

            # glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._frame_texture)
            glUniform1i(self._frame_loc_texture, 0)
            glBindVertexArray(self._frame_vao)
            glDrawElements(GL_TRIANGLES, len(self._frame_indices), GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D, 0)

        
        # Check for OpenGL errors
        check_gl_error()

        self.imgui_renderer.render(imgui.get_draw_data())

        glfw.swap_buffers(self.window)

    def _draw_grid(self):
        glUseProgram(self._grid_shader)
            
        glUniformMatrix4fv(self._loc_grid_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        glUniformMatrix4fv(self._loc_grid_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))
        # glUniformMatrix4fv(self._loc_grid_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))

        glBindVertexArray(self._grid_vao)
        glDrawArrays(GL_LINES, 0, self._grid_vertex_count)
        glBindVertexArray(0)

    def _draw_sky(self):
        glUseProgram(self._sky_shader)

        glUniformMatrix4fv(self._loc_sky_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        glUniformMatrix4fv(self._loc_sky_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))
        # glUniformMatrix4fv(self._loc_sky_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))
        glUniform3f(self._loc_sky_view_pos, *self._camera_pos)
        
        glBindVertexArray(self._sky_vao)
        glDrawElements(GL_TRIANGLES, self._sky_tri_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def _render_scene(self):
        start_instance_idx = 0
        for shape, (vao, vbo, ebo, tri_count) in self._shape_gl_buffers.items():

            num_instances = len(self._shape_instances[shape])

            shape_geo_hash = self._shapes[shape][4]
            if shape_geo_hash == 123 and not self.draw_axis:
                start_instance_idx += num_instances
                continue

            glBindVertexArray(vao)
            # glBindBuffer(GL_ARRAY_BUFFER, vbo)
            # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)

            
            # # Set up vertex attributes
            # vertex_stride = 8 * 4
            # # positions
            # glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(0))
            # glEnableVertexAttribArray(0)
            # # normals
            # glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(3 * 4))
            # glEnableVertexAttribArray(1)
            # # uv coordinates
            # glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(6 * 4))
            # glEnableVertexAttribArray(2)

            # glDrawElementsInstanced(GL_TRIANGLES, tri_count, GL_UNSIGNED_INT, None, num_instances)
            glDrawElementsInstancedBaseInstance(GL_TRIANGLES, tri_count, GL_UNSIGNED_INT, None, num_instances, start_instance_idx)
            # glDrawArrays(GL_TRIANGLES, 0, tri_count)

            start_instance_idx += num_instances

        for instancer in self._shape_instancers.values():
            instancer.render()

        glBindVertexArray(0)

    def _render_scene_tiled(self):
        n = len(self._tile_instances)
        tile_width = self._tile_width
        tile_height = self._tile_height

        i = 0
        for vy in range(self._tile_nrows):
            for vx in range(self._tile_ncols):
                if i >= n:
                    break

                glViewport(vx*tile_width, self.screen_height-(vy+1)*tile_height, tile_width, tile_height)
                if self.draw_grid:
                    self._draw_grid()

                if self.draw_sky:
                    self._draw_sky()
                
                glUseProgram(self._shape_shader)
                
                instances = self._tile_instances[i]

                for instance in instances:
                    shape = self._instance_shape[instance]

                    shape_geo_hash = self._shapes[shape][4]
                    if shape_geo_hash == 123 and not self.draw_axis:
                        # skip axis
                        continue

                    vao, vbo, ebo, tri_count = self._shape_gl_buffers[shape]

                    start_instance_idx = self._inverse_instance_ids[instance]

                    glBindVertexArray(vao)
                    glDrawElementsInstancedBaseInstance(GL_TRIANGLES, tri_count, GL_UNSIGNED_INT, None, 1, start_instance_idx)

                for instancer in self._shape_instancers.values():
                    instancer.render()

                i += 1
        
        glBindVertexArray(0)
    
    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self._left_mouse_pressed = True
                xpos, ypos = glfw.get_cursor_pos(window)
                self._last_x, self._last_y = xpos, ypos
                self._first_mouse = False
            elif action == glfw.RELEASE:
                self._left_mouse_pressed = False

    def _mouse_callback(self, window, xpos, ypos):
        if self._left_mouse_pressed:
            if self._first_mouse:
                self._last_x, self._last_y = xpos, ypos
                self._first_mouse = False
                return

            x_offset = xpos - self._last_x
            y_offset = self._last_y - ypos
            self._last_x, self._last_y = xpos, ypos

            sensitivity = 0.1
            x_offset *= sensitivity
            y_offset *= sensitivity

            self._yaw += x_offset
            self._pitch += y_offset

            self._pitch = max(min(self._pitch, 89.0), -89.0)

            front = glm.vec3()
            front.x = np.cos(np.deg2rad(self._yaw)) * np.cos(np.deg2rad(self._pitch))
            front.y = np.sin(np.deg2rad(self._pitch))
            front.z = np.sin(np.deg2rad(self._yaw)) * np.cos(np.deg2rad(self._pitch))
            self._camera_front = glm.normalize(front)

    def _pressed_key(self, key):
        # only return True when this key has been pressed and now released to avoid flickering toggles
        if glfw.get_key(self.window, key) == glfw.PRESS:
            self._keys_pressed[key] = True
        elif glfw.get_key(self.window, key) == glfw.RELEASE and self._keys_pressed[key]:
            self._keys_pressed[key] = False
            return True
        return False

    def _process_input(self, window):
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS or glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            self._camera_pos += self._camera_speed * self._camera_front * self._frame_speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS or glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            self._camera_pos -= self._camera_speed * self._camera_front * self._frame_speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS or glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            self._camera_pos -= self._camera_speed * glm.normalize(glm.cross(self._camera_front, self._camera_up)) * self._frame_speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            self._camera_pos += self._camera_speed * glm.normalize(glm.cross(self._camera_front, self._camera_up)) * self._frame_speed
        
        if self._pressed_key(glfw.KEY_ESCAPE):
            glfw.set_window_should_close(window, True)
        if self._pressed_key(glfw.KEY_SPACE):
            self.paused = not self.paused
        if self._pressed_key(glfw.KEY_TAB):
            self.skip_rendering = not self.skip_rendering
        if self._pressed_key(glfw.KEY_C):
            self.draw_axis = not self.draw_axis
        if self._pressed_key(glfw.KEY_G):
            self.draw_grid = not self.draw_grid
    
    def _scroll_callback(self, window, x_offset, y_offset):
        self.camera_fov -= y_offset
        self.camera_fov = max(min(self.camera_fov, 90.0), 15.0)
        self.update_projection_matrix()

    def _window_resize_callback(self, window, width, height):
        self._first_mouse = True
        glViewport(0, 0, width, height)
        self.screen_width = width
        self.screen_height = height
        self.update_projection_matrix()
        self._setup_framebuffer()
    
    def register_shape(self, geo_hash, vertices, indices, color1=None, color2=None):
        shape = len(self._shapes)
        if color1 is None:
            color1 = tab10_color_map(len(self._shape_geo_hash))
        if color2 is None:
            color2 = np.clip(np.array(color1) + 0.25, 0.0, 1.0)
        # TODO check if we actually need to store the shape data
        self._shapes.append((vertices, indices, color1, color2, geo_hash))
        self._shape_geo_hash[geo_hash] = shape
        
        glUseProgram(self._shape_shader)

        # Create VAO, VBO, and EBO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.flatten(), GL_STATIC_DRAW)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # normals
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        glEnableVertexAttribArray(1)
        # uv coordinates
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

        self._shape_gl_buffers[shape] = (vao, vbo, ebo, len(indices))

        return shape
    
    def add_shape_instance(self, name: str, shape: int, body, pos, rot, scale=(1.,1.,1.), color1=None, color2=None):
        if color1 is None:
            color1 = self._shapes[shape][2]
        if color2 is None:
            color2 = self._shapes[shape][3]
        instance = len(self._instances)
        self._shape_instances[shape].append(instance)
        body = self._resolve_body_id(body)
        self._instances[name] = (instance, body, shape, [*pos, *rot], scale, color1, color2)
        self._instance_shape[instance] = shape
        self._add_shape_instances = True
        self._instance_count = len(self._instances)
        return instance
    
    def allocate_shape_instances(self):
        self._add_shape_instances = False
        self._wp_instance_transforms = wp.array([instance[3] for instance in self._instances.values()], dtype=wp.transform, device=self._device)
        self._wp_instance_scalings = wp.array([instance[4] for instance in self._instances.values()], dtype=wp.vec3, device=self._device)
        self._wp_instance_bodies = wp.array([instance[1] for instance in self._instances.values()], dtype=wp.int32, device=self._device)

        glUseProgram(self._shape_shader)
        if self._instance_transform_gl_buffer is not None:
            glDeleteBuffers(1, [self._instance_transform_gl_buffer])
            glDeleteBuffers(1, [self._instance_color1_buffer])
            glDeleteBuffers(1, [self._instance_color2_buffer])
        
        # Create instance buffer and bind it as an instanced array
        self._instance_transform_gl_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._instance_transform_gl_buffer)

        transforms = np.tile(np.diag(np.ones(4, dtype=np.float32)), (len(self._instances), 1, 1))
        glBufferData(GL_ARRAY_BUFFER, transforms.nbytes, transforms, GL_DYNAMIC_DRAW)
        # glVertexAttribDivisor(1, vertices.shape[0])

        # Create CUDA buffer for instance transforms
        self._instance_transform_cuda_buffer = pycuda.gl.RegisteredBuffer(int(self._instance_transform_gl_buffer))

        # Upload instance matrices to GPU
        # offset = 0
        # for matrix in transforms:
        #     glBufferSubData(GL_ARRAY_BUFFER, offset, matrix.nbytes, matrix)
        #     offset += matrix.nbytes

        colors1, colors2 = [], []
        all_instances = list(self._instances.values())
        for shape, instances in self._shape_instances.items():
            for i in instances:
                instance = all_instances[i]
                colors1.append(instance[5])
                colors2.append(instance[6])
        colors1 = np.array(colors1, dtype=np.float32)
        colors2 = np.array(colors2, dtype=np.float32)

        # colors1 = np.array([instance[5] for instance in self._instances.values()], dtype=np.float32)
        # colors2 = np.array([instance[6] for instance in self._instances.values()], dtype=np.float32)

        # create buffer for checkerboard colors
        self._instance_color1_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._instance_color1_buffer)
        glBufferData(GL_ARRAY_BUFFER, colors1.nbytes, colors1.flatten(), GL_STATIC_DRAW)
        
        self._instance_color2_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._instance_color2_buffer)
        glBufferData(GL_ARRAY_BUFFER, colors2.nbytes, colors2.flatten(), GL_STATIC_DRAW)

        # Set up instance attribute pointers
        matrix_size = transforms[0].nbytes

        instance_ids = []
        inverse_instance_ids = {}
        instance_count = 0
        for shape, (vao, vbo, ebo, tri_count) in self._shape_gl_buffers.items():
            glBindVertexArray(vao)

            glBindBuffer(GL_ARRAY_BUFFER, self._instance_transform_gl_buffer)

            # glVertexAttribPointer(3, 4*4, GL_FLOAT, GL_FALSE, matrix_size, ctypes.c_void_p(0))
            # glEnableVertexAttribArray(3)
            # we can only send vec4s to the shader, so we need to split the instance transforms matrix into its column vectors
            for i in range(4):
                glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, matrix_size, ctypes.c_void_p(i * matrix_size // 4))
                glEnableVertexAttribArray(3 + i)
                glVertexAttribDivisor(3 + i, 1)

            glBindBuffer(GL_ARRAY_BUFFER, self._instance_color1_buffer)
            glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, colors1[0].nbytes, ctypes.c_void_p(0))
            glEnableVertexAttribArray(7)
            glVertexAttribDivisor(7, 1)
        
            glBindBuffer(GL_ARRAY_BUFFER, self._instance_color2_buffer)
            glVertexAttribPointer(8, 3, GL_FLOAT, GL_FALSE, colors2[0].nbytes, ctypes.c_void_p(0))
            glEnableVertexAttribArray(8)
            glVertexAttribDivisor(8, 1)

            instance_ids.extend(self._shape_instances[shape])
            for i in self._shape_instances[shape]:
                inverse_instance_ids[i] = instance_count
                instance_count += 1
        
        # trigger update to the instance transforms
        self._update_shape_instances = True

        self._wp_instance_ids = wp.array(instance_ids, dtype=wp.int32, device=self._device)
        self._instance_ids = instance_ids
        self._inverse_instance_ids = inverse_instance_ids

        glBindVertexArray(0)
   
    def update_shape_instance(self, name, pos, rot, color1=None, color2=None):
        """Update the instance transform of the shape
        
        Args:
            name: The name of the shape
            pos: The position of the shape
            rot: The rotation of the shape
        """
        if name in self._instances:
            i, body, shape, _, scale, old_color1, old_color2 = self._instances[name]
            self._instances[name] = (i, body, shape, [*pos, *rot], scale, color1 or old_color1, color2 or old_color2)
            self._update_shape_instances = True
            return True
        return False
    
    def update_shape_instances(self):
        glUseProgram(self._shape_shader)

        self._update_shape_instances = False
        self._wp_instance_transforms = wp.array([instance[3] for instance in self._instances.values()], dtype=wp.transform, device=self._device)
        self.update_body_transforms(None)

    def update_body_transforms(self, body_tf: wp.array):
        if self._instance_transform_cuda_buffer is None:
            return
        
        body_q = None
        if body_tf is not None:
            if body_tf.device.is_cuda:
                body_q = body_tf
            else:
                body_q = body_tf.to(self._device)
        
        mapped_buffer = self._instance_transform_cuda_buffer.map()
        ptr, _ = mapped_buffer.device_ptr_and_size()
        vbo_transforms = wp.array(dtype=wp.mat44, shape=(self._instance_count,), device=self._device, ptr=ptr, owner=False)

        wp.launch(
            update_vbo_transforms,
            dim=self._instance_count,
            inputs=[
                self._wp_instance_ids,
                self._wp_instance_bodies,
                self._wp_instance_transforms,
                self._wp_instance_scalings,
                body_q,
            ],
            outputs=[
                vbo_transforms,
            ],
            device=self._device)
        
        mapped_buffer.unmap()

    def register_body(self, name):
        # register body name and return its ID
        if name not in self._body_name:
            self._body_name[name] = len(self._body_name)
        return self._body_name[name]

    def _resolve_body_id(self, body):
        if body is None:
            return -1
        if isinstance(body, int):
            return body
        return self._body_name[body]

    def is_running(self):
        return not glfw.window_should_close(self.window)

    def save(self):
        # save just keeps the window open to allow the user to interact with the scene
        while not glfw.window_should_close(self.window):
            self.update()
        if glfw.window_should_close(self.window):
            self.clear()
            glfw.terminate()

    def get_pixels(self, target_image: wp.array, split_up_tiles=True):
        split_up_tiles = split_up_tiles and self._tile_ncols > 0
        if split_up_tiles:
            assert target_image.shape == (self.num_tiles, self._tile_height, self._tile_width, 3), f"Shape of `target_image` array does not match {self.num_tiles} x {self.screen_height} x {self.screen_width} x 3"
        else:
            assert target_image.shape == (self.screen_height, self.screen_width, 3), f"Shape of `target_image` array does not match {self.screen_height} x {self.screen_width} x 3"

        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._frame_pbo)
        glBindTexture(GL_TEXTURE_2D, self._frame_texture)
        # read screen texture into PBO
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, ctypes.c_void_p(0))
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

        pbo_buffer = pycuda.gl.RegisteredBuffer(int(self._frame_pbo), pycuda.gl.graphics_map_flags.WRITE_DISCARD)
        mapped_buffer = pbo_buffer.map()
        ptr, _ = mapped_buffer.device_ptr_and_size()
        num_tiles = max(1, self.num_tiles)
        img = wp.array(dtype=wp.float32, shape=(num_tiles*self._tile_width*self._tile_height*3), device=self._device, ptr=ptr, owner=False)
        if split_up_tiles:
            wp.launch(
                copy_frame_tiles,
                dim=(self.num_tiles, self._tile_width, self._tile_height),
                inputs=[
                    img,
                    self.screen_width,
                    self._tile_ncols,
                    self._tile_width,
                    self._tile_height
                ],
                outputs=[target_image]
            )
        else:
            wp.launch(
                copy_frame,
                dim=(self.screen_width, self.screen_height),
                inputs=[
                    img,
                    self.screen_width,
                    self.screen_height
                ],
                outputs=[target_image])
        mapped_buffer.unmap()

    # def create_image_texture(self, file_path):
    #     from PIL import Image
    #     img = Image.open(file_path)
    #     img_data = np.array(list(img.getdata()), np.uint8)
    #     texture = glGenTextures(1)
    #     glBindTexture(GL_TEXTURE_2D, texture)
    #     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    #     return texture

    # def create_check_texture(self, color1=(0, 0.5, 1.0), color2=None, width=default_texture_size, height=default_texture_size):
    #     if width == 1 and height == 1:        
    #         pixels = np.array([np.array(color1)*255], dtype=np.uint8)
    #     else:
    #         pixels = np.zeros((width, height, 3), dtype=np.uint8)
    #         half_w = width // 2
    #         half_h = height // 2
    #         color1 = np.array(np.array(color1)*255, dtype=np.uint8)
    #         pixels[0:half_w, 0:half_h] = color1
    #         pixels[half_w:width, half_h:height] = color1
    #         if color2 is None:
    #             color2 = np.array(np.clip(np.array(color1, dtype=np.float32) + 50, 0, 255), dtype=np.uint8)
    #         else:
    #             color2 = np.array(np.array(color2)*255, dtype=np.uint8)
    #         pixels[half_w:width, 0:half_h] = color2
    #         pixels[0:half_w, half_h:height] = color2
    #     texture = glGenTextures(1)
    #     glBindTexture(GL_TEXTURE_2D, texture)
    #     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.flatten())
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    #     return texture

    def render_plane(self, name: str, pos: tuple, rot: tuple, width: float, length: float, color: tuple=(1.,1.,1.), color2=None, parent_body: str=None, is_template: bool=False, u_scaling=1.0, v_scaling=1.0):
        """Add a plane for visualization
        
        Args:
            name: The name of the plane
            pos: The position of the plane
            rot: The rotation of the plane
            width: The width of the plane
            length: The length of the plane
            color: The color of the plane
            texture: The texture of the plane (optional)
        """
        geo_hash = hash(("plane", width, length))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            faces = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            normal = (0.0, 1.0, 0.0)
            width = (width if width > 0.0 else 100.0)
            length = (length if length > 0.0 else 100.0)
            aspect = width / length
            u = width * aspect * u_scaling
            v = length * v_scaling
            gfx_vertices = np.array([
                [-width, 0.0, -length, *normal, 0.0, 0.0],
                [-width, 0.0,  length, *normal, 0.0, v],
                [width, 0.0,  length, *normal, u, v],
                [width, 0.0, -length, *normal, u, 0.0],
            ], dtype=np.float32)
            shape = self.register_shape(geo_hash, gfx_vertices, faces, color1=color, color2=color2)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_ground(self, size: float=100.0):
        """Add a ground plane for visualization
        
        Args:
            size: The size of the ground plane
        """
        color1 = (200/255, 200/255, 200/255)
        color2 = (150/255, 150/255, 150/255)
        return self.render_plane("ground", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), size, size, color1, color2=color2, u_scaling=1.0, v_scaling=1.0)
        
    def render_sphere(self, name: str, pos: tuple, rot: tuple, radius: float, parent_body: str=None, is_template: bool=False):
        """Add a sphere for visualization
        
        Args:
            pos: The position of the sphere
            radius: The radius of the sphere
            name: A name for the USD prim on the stage
        """
        geo_hash = hash(("sphere", radius))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_sphere_mesh(radius)
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape
    
    def render_capsule(self, name: str, pos: tuple, rot: tuple, radius: float, half_height: float, parent_body: str=None, is_template: bool=False):
        """Add a capsule for visualization
        
        Args:
            pos: The position of the capsule
            radius: The radius of the capsule
            half_height: The half height of the capsule
            name: A name for the USD prim on the stage
        """
        geo_hash = hash(("capsule", radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_capsule_mesh(radius, half_height)
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape
    
    def render_cylinder(self, name: str, pos: tuple, rot: tuple, radius: float, half_height: float, parent_body: str=None, is_template: bool=False):
        """Add a cylinder for visualization
        
        Args:
            pos: The position of the cylinder
            radius: The radius of the cylinder
            half_height: The half height of the cylinder
            name: A name for the USD prim on the stage
        """
        geo_hash = hash(("cylinder", radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_cylinder_mesh(radius, half_height)
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape
    
    def render_cone(self, name: str, pos: tuple, rot: tuple, radius: float, half_height: float, parent_body: str=None, is_template: bool=False):
        """Add a cone for visualization
        
        Args:
            pos: The position of the cone
            radius: The radius of the cone
            half_height: The half height of the cone
            name: A name for the USD prim on the stage
        """
        geo_hash = hash(("cone", radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_cone_mesh(radius, half_height)
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape
    
    def render_box(self, name: str, pos: tuple, rot: tuple, extents: tuple, parent_body: str=None, is_template: bool=False):
        """Add a box for visualization
        
        Args:
            pos: The position of the box
            extents: The extents of the box
            name: A name for the USD prim on the stage
        """
        geo_hash = hash(("box", tuple(extents)))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_box_mesh(extents)
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape
    
    def render_mesh(self, name: str, points, indices, colors=None, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), scale=(1.0, 1.0, 1.0), update_topology=False, parent_body: str=None, is_template: bool=False, smooth_shading: bool=True):
        """Add a mesh for visualization
        
        Args:
            points: The points of the mesh
            indices: The indices of the mesh
            colors: The colors of the mesh
            pos: The position of the mesh
            rot: The rotation of the mesh
            scale: The scale of the mesh
            name: A name for the USD prim on the stage
            smooth_shading: Whether to average face normals at each vertex or introduce additional vertices for each face
        """
        if colors is None:
            colors = np.ones((len(points), 3), dtype=np.float32)
        else:
            colors = np.array(colors, dtype=np.float32)
        points = np.array(points, dtype=np.float32) * np.array(scale, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32).reshape((-1, 3))
        geo_hash = hash((points.tobytes(), indices.tobytes(), colors.tobytes()))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            if smooth_shading:
                normals = wp.zeros(len(points), dtype=wp.vec3)
                vertices = wp.array(points, dtype=wp.vec3)
                faces_per_vertex = wp.zeros(len(points), dtype=int)
                wp.launch(
                    compute_average_normals,
                    dim=len(indices),
                    inputs=[
                        wp.array(indices, dtype=int),
                        vertices
                    ],
                    outputs=[
                        normals,
                        faces_per_vertex
                    ])
                gfx_vertices = wp.zeros((len(points), 8), dtype=float)
                wp.launch(
                    assemble_gfx_vertices,
                    dim=len(points),
                    inputs=[
                        vertices,
                        normals,
                        faces_per_vertex
                    ],
                    outputs=[gfx_vertices])
                gfx_vertices = gfx_vertices.numpy()
                gfx_indices = indices.flatten()
            else:
                gfx_vertices = wp.zeros((len(indices)*3, 8), dtype=float)
                wp.launch(
                    compute_gfx_vertices,
                    dim=len(indices),
                    inputs=[
                        wp.array(indices, dtype=int),
                        wp.array(points, dtype=wp.vec3)
                    ],
                    outputs=[gfx_vertices])
                gfx_vertices = gfx_vertices.numpy()
                gfx_indices = np.arange(len(indices)*3)
            shape = self.register_shape(geo_hash, gfx_vertices, gfx_indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape
    
    def render_points(self, name: str, points, radius, colors=None):
        """Add a set of points
        
        Args:
            points: The points to render
            radius: The radius of the points
            colors: The colors of the points
            name: A name for the USD prim on the stage
        """

        if len(points) == 0:
            return
        
        if isinstance(points, wp.array):
            wp_points = points
        else:
            wp_points = wp.array(points, dtype=wp.vec3, device=self._device)
        
        if name not in self._shape_instancers:
            instancer = ShapeInstancer(self._shape_shader, self._device)
            vertices, indices = self._create_sphere_mesh(radius)
            if colors is None:
                color = tab10_color_map(len(self._shape_geo_hash))
            else:
                color = colors[0]
            instancer.register_shape(vertices, indices, color, color)
            instancer.allocate_instances(np.array(points))
            self._shape_instancers[name] = instancer
        else:
            instancer = self._shape_instancers[name]

        with instancer:
            wp.launch(
                update_points_positions,
                dim=len(points),
                inputs=[wp_points, None],
                outputs=[instancer.vbo_transforms],
                device=self._device)

    @staticmethod
    def _create_sphere_mesh(radius=1.0, num_latitudes=default_num_segments, num_longitudes=default_num_segments):
        vertices = []
        indices = []

        for i in range(num_latitudes + 1):
            theta = i * np.pi / num_latitudes
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(num_longitudes + 1):
                phi = j * 2 * np.pi / num_longitudes
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                x = cos_phi * sin_theta
                y = cos_theta
                z = sin_phi * sin_theta

                u = float(j) / num_longitudes
                v = float(i) / num_latitudes

                vertices.append([x * radius, y * radius, z * radius, x, y, z, u, v])

        for i in range(num_latitudes):
            for j in range(num_longitudes):
                first = i * (num_longitudes + 1) + j
                second = first + num_longitudes + 1

                indices.extend([first, second, first + 1, second, second + 1, first + 1])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)
    
    @staticmethod
    def _create_capsule_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
        vertices = []
        indices = []

        x_dir, y_dir, z_dir = (
            (1, 2, 0),
            (2, 0, 1),
            (0, 1, 2)
        )[up_axis]
        up_vector = np.zeros(3)
        up_vector[up_axis] = half_height

        for i in range(segments + 1):
            theta = i * np.pi / segments
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(segments + 1):
                phi = j * 2 * np.pi / segments
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                z = cos_phi * sin_theta
                y = cos_theta
                x = sin_phi * sin_theta

                u = cos_theta * 0.5 + 0.5
                v = cos_phi * sin_theta * 0.5 + 0.5

                xyz = x, y, z
                x, y, z = xyz[x_dir], xyz[y_dir], xyz[z_dir]
                xyz = np.array((x, y, z), dtype=np.float32) * radius
                if j < segments // 2:
                    xyz += up_vector
                else:
                    xyz -= up_vector

                vertices.append([*xyz, x, y, z, u, v])

        nv = len(vertices)
        for i in range(segments+1):
            for j in range(segments+1):
                first = (i * (segments + 1) + j) % nv
                second = (first + segments + 1) % nv
                indices.extend([first, second, (first + 1) % nv, second, (second + 1) % nv, (first + 1) % nv])

        vertex_data = np.array(vertices, dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)

        return vertex_data, index_data
    
    @staticmethod
    def _create_cone_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
        # render it as a cylinder with zero top radius so we get correct normals on the sides
        return TinyRenderer._create_cylinder_mesh(radius, half_height, up_axis, segments, 0.0)
    
    @staticmethod
    def _create_cylinder_mesh(radius, half_height, up_axis=1, segments=default_num_segments, top_radius=None):
        if up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be between 0 and 2")
        
        x_dir, y_dir, z_dir = (
            (1, 2, 0),
            (0, 1, 2),
            (2, 0, 1),
        )[up_axis]

        indices = []

        cap_vertices = []
        side_vertices = []

        # create center cap vertices
        position = np.array([0, -half_height, 0])[[x_dir, y_dir, z_dir]]
        normal = np.array([0, -1, 0])[[x_dir, y_dir, z_dir]]
        cap_vertices.append([*position, *normal, 0.5, 0.5])
        cap_vertices.append([*-position, *-normal, 0.5, 0.5])

        if top_radius is None:
            top_radius = radius
        side_slope = -np.arctan2(top_radius-radius, 2*half_height)

        # Create the cylinder base and top vertices
        for j in (-1, 1):
            center_index = max(j, 0)
            if j == 1:
                radius = top_radius
            for i in range(segments):
                theta = 2 * np.pi * i / segments

                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                x = cos_theta
                y = j * half_height
                z = sin_theta

                position = np.array([radius * x, y, radius * z])

                normal = np.array([x, side_slope, z])
                normal = normal / np.linalg.norm(normal)
                uv = (i / (segments-1), (j + 1) / 2)
                vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
                side_vertices.append(vertex)

                normal = np.array([0, j, 0])
                uv = (cos_theta*0.5+0.5, sin_theta*0.5+0.5)
                vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
                cap_vertices.append(vertex)

                indices.extend([center_index, i+center_index*segments+2, (i+1)%segments+center_index*segments+2])

        # Create the cylinder side indices
        for i in range(segments):
            index1 = len(cap_vertices) + i + segments
            index2 = len(cap_vertices) + ((i + 1) % segments) + segments
            index3 = len(cap_vertices) + i
            index4 = len(cap_vertices) + ((i + 1) % segments)

            indices.extend([index1, index2, index3, index2, index4, index3])

        vertex_data = np.array(np.vstack((cap_vertices, side_vertices)), dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)

        return vertex_data, index_data
    
    @staticmethod
    def _create_arrow_mesh(base_radius, base_height, cap_radius=None, cap_height=None, up_axis=1, segments=default_num_segments):
        if up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be between 0 and 2")
        if cap_radius is None:
            cap_radius = base_radius * 1.8
        if cap_height is None:
            cap_height = base_height * 0.18

        up_vector = np.array([0, 0, 0])
        up_vector[up_axis] = 1

        base_vertices, base_indices = TinyRenderer._create_cylinder_mesh(base_radius, base_height/2, up_axis, segments)
        cap_vertices, cap_indices = TinyRenderer._create_cone_mesh(cap_radius, cap_height/2, up_axis, segments)

        base_vertices[:,:3] += base_height/2 * up_vector
        # move cap slightly lower to avoid z-fighting
        cap_vertices[:,:3] += (base_height + cap_height/2 - 1e-3*base_height) * up_vector

        vertex_data = np.vstack((base_vertices, cap_vertices))
        index_data = np.hstack((base_indices, cap_indices + len(base_vertices)))

        return vertex_data, index_data
    
    @staticmethod
    def _create_box_mesh(extents):
        x_extent, y_extent, z_extent = extents

        vertices = [
            # Position                        Normal    UV
            [-x_extent, -y_extent, -z_extent, -1, 0, 0, 0, 0],
            [-x_extent, -y_extent,  z_extent, -1, 0, 0, 1, 0],
            [-x_extent,  y_extent,  z_extent, -1, 0, 0, 1, 1],
            [-x_extent,  y_extent, -z_extent, -1, 0, 0, 0, 1],

            [x_extent, -y_extent, -z_extent, 1, 0, 0, 0, 0],
            [x_extent, -y_extent,  z_extent, 1, 0, 0, 1, 0],
            [x_extent,  y_extent,  z_extent, 1, 0, 0, 1, 1],
            [x_extent,  y_extent, -z_extent, 1, 0, 0, 0, 1],

            [-x_extent, -y_extent, -z_extent, 0, -1, 0, 0, 0],
            [-x_extent, -y_extent,  z_extent, 0, -1, 0, 1, 0],
            [ x_extent, -y_extent,  z_extent, 0, -1, 0, 1, 1],
            [ x_extent, -y_extent, -z_extent, 0, -1, 0, 0, 1],

            [-x_extent,  y_extent, -z_extent, 0, 1, 0, 0, 0],
            [-x_extent,  y_extent,  z_extent, 0, 1, 0, 1, 0],
            [ x_extent,  y_extent,  z_extent, 0, 1, 0, 1, 1],
            [ x_extent,  y_extent, -z_extent, 0, 1, 0, 0, 1],

            [-x_extent, -y_extent, -z_extent, 0, 0, -1, 0, 0],
            [-x_extent,  y_extent, -z_extent, 0, 0, -1, 1, 0],
            [ x_extent,  y_extent, -z_extent, 0, 0, -1, 1, 1],
            [ x_extent, -y_extent, -z_extent, 0, 0, -1, 0, 1],

            [-x_extent, -y_extent,  z_extent, 0, 0, 1, 0, 0],
            [-x_extent,  y_extent,  z_extent, 0, 0, 1, 1, 0],
            [ x_extent,  y_extent,  z_extent, 0, 0, 1, 1, 1],
            [ x_extent, -y_extent,  z_extent, 0, 0, 1, 0, 1],
        ]

        indices = [
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
            8, 9, 10, 8, 10, 11,
            12, 13, 14, 12, 14, 15,
            16, 17, 18, 16, 18, 19,
            20, 21, 22, 20, 22, 23
        ]
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)
    

if __name__ == "__main__":
    wp.init()
    renderer = TinyRenderer()
