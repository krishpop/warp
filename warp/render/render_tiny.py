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
# from .utils import mul_elemwise, tab10_color_map

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


vertex_shader = '''
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
uniform mat4 projection;

out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoord;
out vec3 ObjectColor1;
out vec3 ObjectColor2;

void main()
{
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1, aInstanceTransform2, aInstanceTransform3);
    vec4 worldPos = model * vec4(aPos, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = vec3(worldPos);
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    ObjectColor1 = aObjectColor1;
    ObjectColor2 = aObjectColor2;
}
'''

fragment_shader = '''
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord;
in vec3 ObjectColor1;
in vec3 ObjectColor2;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

void main()
{
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    // Checkerboard pattern
    float u = TexCoord.x;
    float v = TexCoord.y;
    float scale = 8.0;
    float checker = mod(floor(u * scale) + floor(v * scale), 2.0);

    // Set the colors for the checkerboard pattern
    vec3 checkerColor = mix(ObjectColor1, ObjectColor2, checker);

    vec3 result = (ambient + diffuse + specular) * checkerColor;
    FragColor = vec4(result, 1.0);
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

def check_gl_error():
    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        print(f"OpenGL error: {error}")

def check_shader_error(shader):
    success = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
    if not success:
        info_log = gl.glGetShaderInfoLog(shader)
        print(f"Shader compilation error: {info_log}")

def check_program_error(program):
    success = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not success:
        info_log = gl.glGetProgramInfoLog(program)
        print(f"Shader linking error: {info_log}")


class TinyRenderer:
    # number of segments to use for rendering spheres, capsules, cones and cylinders
    default_num_segments = 32

    # number of horizontal and vertical pixels to use for checkerboard texture
    default_texture_size = 256

    def __init__(self, near_plane=0.001, far_plane=1000.0, camera_fov=45.0, background_color=(1., 1., 1.)):
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.camera_fov = camera_fov
        self.background_color = background_color

        if not glfw.init():
            raise Exception("GLFW initialization failed!")

        self.window = glfw.create_window(800, 600, "OpenGL Instancing Example - Spheres", None, None)

        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed!")
        
        self._camera_pos = glm.vec3(0.0, 0.0, 10.0)
        self._camera_front = glm.vec3(0.0, 0.0, -1.0)
        self._camera_up = glm.vec3(0.0, 1.0, 0.0)
        self._camera_speed = 0.1
        self._yaw, self._pitch = -90.0, 0.0
        self._last_x, self._last_y = 800 // 2, 600 // 2
        self._first_mouse = True
        self._left_mouse_pressed = False
        
        # glfw.set_window_pos(self.window, 400, 200)
        glfw.set_window_size_callback(self.window, self._window_resize_callback)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        glfw.make_context_current(self.window)
        
        glClearColor(*self.background_color, 1)
        glEnable(GL_DEPTH_TEST)

        # sphere_vertices, sphere_indices = self._create_sphere_mesh()
        # sphere_vertices, sphere_indices = self._create_capsule_mesh(radius=0.2, half_height=0.5)
        # sphere_vertices, sphere_indices = self._create_cone_mesh(radius=0.2, half_height=0.5)
        sphere_vertices, sphere_indices = self._create_cylinder_mesh(radius=0.2, half_height=0.5)
        # sphere_vertices, sphere_indices = self._create_box_mesh([1.0, 2.0, 3.0])
        num_instances = 50
        instance_positions = np.random.rand(num_instances, 3) * 10 - 5
        instance_colors1 = np.random.rand(num_instances, 3)
        instance_colors2 = np.clip(instance_colors1 + 0.25, 0.0, 1.0)
        instance_colors1 = np.array(instance_colors1, dtype=np.float32)
        instance_colors2 = np.array(instance_colors2, dtype=np.float32)

        sphere_transforms = []

        # Create transform matrices for all spheres
        for i in range(num_instances):
            angle = np.deg2rad(36 * i)
            axis = glm.vec3(0, 1, 0)
            transform = glm.translate(glm.rotate(glm.mat4(1.0), angle, axis), glm.vec3(*instance_positions[i]))
            # transform = glm.mat4(1.0)
            sphere_transforms.append(np.array(transform).T)

        sphere_transforms = np.array(sphere_transforms, dtype=np.float32)

        shader = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
        # Check for shader compilation errors
        # check_shader_error(vertex_shader)

        glUseProgram(shader)

        self._loc_model = glGetUniformLocation(shader, "model")
        self._loc_view = glGetUniformLocation(shader, "view")
        self._loc_projection = glGetUniformLocation(shader, "projection")
        
        # Create VAO, VBO, and EBO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, sphere_vertices.nbytes, sphere_vertices.flatten(), GL_STATIC_DRAW)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.nbytes, sphere_indices, GL_STATIC_DRAW)

        # Set up vertex attributes
        vertex_stride = sphere_vertices.shape[1] * sphere_vertices.itemsize
        # positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # normals
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(3 * sphere_vertices.itemsize))
        glEnableVertexAttribArray(1)
        # uv coordinates
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(6 * sphere_vertices.itemsize))
        glEnableVertexAttribArray(2)

        # Create instance buffer and bind it as an instanced array
        instance_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, instance_buffer)
        glBufferData(GL_ARRAY_BUFFER, sphere_transforms.nbytes, sphere_transforms, GL_DYNAMIC_DRAW)
        # glVertexAttribDivisor(1, sphere_vertices.shape[0])

        # Upload instance matrices to GPU
        # offset = 0
        # for matrix in sphere_transforms:
        #     glBufferSubData(GL_ARRAY_BUFFER, offset, matrix.nbytes, matrix)
        #     offset += matrix.nbytes

        # Set up instance attribute pointers
        matrix_size = sphere_transforms[0].nbytes
        # glVertexAttribPointer(3, 4*4, GL_FLOAT, GL_FALSE, matrix_size, ctypes.c_void_p(0))
        # glEnableVertexAttribArray(3)
        # we can only send vec4s to the shader, so we need to split the instance transforms matrix into its column vectors
        for i in range(4):
            glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, matrix_size, ctypes.c_void_p(i * matrix_size // 4))
            glEnableVertexAttribArray(3 + i)
            glVertexAttribDivisor(3 + i, 1)

        # create buffer for checkerboard colors
        color1_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, color1_buffer)
        glBufferData(GL_ARRAY_BUFFER, instance_colors1.nbytes, instance_colors1.flatten(), GL_STATIC_DRAW)
        glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, instance_colors1[0].nbytes, ctypes.c_void_p(0))
        glEnableVertexAttribArray(7)
        glVertexAttribDivisor(7, 1)
        
        color2_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, color2_buffer)
        glBufferData(GL_ARRAY_BUFFER, instance_colors2.nbytes, instance_colors2.flatten(), GL_STATIC_DRAW)
        glVertexAttribPointer(8, 3, GL_FLOAT, GL_FALSE, instance_colors2[0].nbytes, ctypes.c_void_p(0))
        glEnableVertexAttribArray(8)
        glVertexAttribDivisor(8, 1)

        width, height = glfw.get_window_size(self.window)
        projection = glm.perspective(np.deg2rad(45), width / height, self.near_plane, self.far_plane)
        glUniformMatrix4fv(self._loc_projection, 1, GL_FALSE, glm.value_ptr(projection))

        view = glm.lookAt(self._camera_pos, self._camera_pos + self._camera_front, self._camera_up)
        glUniformMatrix4fv(self._loc_view, 1, GL_FALSE, glm.value_ptr(view))
        
        model = glm.mat4(1.0)
        glUniformMatrix4fv(self._loc_model, 1, GL_FALSE, glm.value_ptr(model))

        glUniform3f(glGetUniformLocation(shader, "lightPos"), 5, 5, 5)
        glUniform3f(glGetUniformLocation(shader, "lightColor"), 1, 1, 1)
        glUniform3f(glGetUniformLocation(shader, "viewPos"), 0, 0, 10)

        import pycuda.gl.autoinit
        instance_buffer_cuda = pycuda.gl.BufferObject(int(instance_buffer))
        mapped_buffer = instance_buffer_cuda.map()
        ptr = mapped_buffer.device_ptr()
        mapped_buffer.unmap()

        wp_positions = wp.array(instance_positions, dtype=wp.vec3, device='cuda')
        wp_scalings = wp.array(np.random.uniform(0.5, 2.0, size=(num_instances, 3)), dtype=wp.vec3, device='cuda')

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self._process_input(self.window)

            glClearColor(*self.background_color, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            view = glm.lookAt(self._camera_pos, self._camera_pos + self._camera_front, self._camera_up)
            glUniformMatrix4fv(self._loc_view, 1, GL_FALSE, glm.value_ptr(view))

            glUniform3f(glGetUniformLocation(shader, "viewPos"), *self._camera_pos)

            glBindVertexArray(vao)


            
            if False:
                sphere_transforms = []
                # Create transform matrices for all spheres
                for i in range(num_instances):
                    angle = np.deg2rad(36 * i) + glfw.get_time()
                    axis = glm.vec3(0, 1, 0)
                    offset = glm.vec3(0.0, 0.0, 5.0 * np.sin(i*np.pi/4 + glfw.get_time()))
                    # transform = glm.translate(glm.rotate(glm.mat4(1.0), angle, axis), glm.vec3(*instance_positions[i])+offset)
                    transform = glm.rotate(glm.translate(glm.mat4(1.0), glm.vec3(*instance_positions[i])+offset), angle, axis)
                    transform = glm.scale(transform, glm.vec3(0.25, 0.25, 0.25))
                    # transform = glm.mat4(1.0)
                    sphere_transforms.append(np.array(transform).T)
                sphere_transforms = np.array(sphere_transforms, dtype=np.float32)
                glBindBuffer(GL_ARRAY_BUFFER, instance_buffer)
                glBufferData(GL_ARRAY_BUFFER, sphere_transforms.nbytes, sphere_transforms, GL_DYNAMIC_DRAW)
            else:
                mapped_buffer = instance_buffer_cuda.map()
                ptr = mapped_buffer.device_ptr()
                wp_instance_buffer = wp.array(dtype=wp.mat44, shape=(num_instances,), device='cuda', ptr=ptr, owner=False)
                wp.launch(
                    move_instances,
                    dim=num_instances,
                    inputs=[wp_positions, wp_scalings, glfw.get_time()],
                    outputs=[wp_instance_buffer],
                    device=wp_instance_buffer.device
                )
                mapped_buffer.unmap()


            glDrawElementsInstanced(GL_TRIANGLES, len(sphere_indices), GL_UNSIGNED_INT, None, num_instances)

            
            # Check for OpenGL errors
            check_gl_error()

            glfw.swap_buffers(self.window)

        # Clean up
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])
        glDeleteBuffers(1, [ebo])
        glDeleteBuffers(1, [instance_buffer])
        glfw.terminate()

    def update_projection_matrix(self):
        resolution = glfw.get_framebuffer_size(self.window)
        if resolution[1] == 0:
            return
        aspect_ratio = resolution[0] / resolution[1]
        projection = glm.perspective(glm.radians(self.camera_fov), aspect_ratio, self.near_plane, self.far_plane)
        glUniformMatrix4fv(self._loc_projection, 1, GL_FALSE, glm.value_ptr(projection))
    
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

    def _process_input(self, window):
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self._camera_pos += self._camera_speed * self._camera_front
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self._camera_pos -= self._camera_speed * self._camera_front
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self._camera_pos -= self._camera_speed * glm.normalize(glm.cross(self._camera_front, self._camera_up))
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self._camera_pos += self._camera_speed * glm.normalize(glm.cross(self._camera_front, self._camera_up))
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
    
    def _scroll_callback(self, window, x_offset, y_offset):
        self.camera_fov -= y_offset
        self.camera_fov = max(min(self.camera_fov, 90.0), 15.0)
        self.update_projection_matrix()

    def _window_resize_callback(self, window, width, height):
        self._first_mouse = True
        glViewport(0, 0, width, height)
        self.update_projection_matrix()
    
    def create_image_texture(self, file_path):
        from PIL import Image
        img = Image.open(file_path)
        img_data = np.array(list(img.getdata()), np.uint8)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return texture

    def create_check_texture(self, color1=(0, 0.5, 1.0), color2=None, width=default_texture_size, height=default_texture_size):
        if width == 1 and height == 1:        
            pixels = np.array([np.array(color1)*255], dtype=np.uint8)
        else:
            pixels = np.zeros((width, height, 3), dtype=np.uint8)
            half_w = width // 2
            half_h = height // 2
            color1 = np.array(np.array(color1)*255, dtype=np.uint8)
            pixels[0:half_w, 0:half_h] = color1
            pixels[half_w:width, half_h:height] = color1
            if color2 is None:
                color2 = np.array(np.clip(np.array(color1, dtype=np.float32) + 50, 0, 255), dtype=np.uint8)
            else:
                color2 = np.array(np.array(color2)*255, dtype=np.uint8)
            pixels[half_w:width, 0:half_h] = color2
            pixels[0:half_w, half_h:height] = color2
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.flatten())
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return texture

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

                u = float(j) / segments / 2
                v = float(i) / segments

                xyz = x, y, z
                x, y, z = xyz[x_dir], xyz[y_dir], xyz[z_dir]
                xyz = np.array((x, y, z), dtype=np.float32) * radius
                if j < segments // 2:
                    xyz += up_vector
                    u += 0.5
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
        if up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be between 0 and 2")

        vertices = []
        indices = []

        h = 2*half_height
        cone_angle = np.arctan2(radius, h)
        cos_angle = np.cos(cone_angle)
        sin_angle = np.sin(cone_angle)

        # Create the cone side vertices
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x_dir, y_dir, z_dir = (
                (1, 2, 0),
                (2, 0, 1),
                (0, 1, 2)
            )[up_axis]

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            x = radius * cos_theta
            y = radius * sin_theta
            z = -half_height

            position = np.array([x, y, z])
            normal = np.array([cos_angle*cos_theta, cos_angle*sin_theta, sin_angle])
            uv = (i / segments, 0)

            vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
            vertices.append(vertex)

        # Create the cone tip vertex
        position = np.array([0, 0, half_height])[[x_dir, y_dir, z_dir]]
        normal = np.array([0, 0, 1])[[x_dir, y_dir, z_dir]]
        vertices.append([*position, *normal, 0.5, 1])

        # Create the cone side indices
        for i in range(segments):
            index1 = i
            index2 = (i + 1) % segments
            index3 = segments
            indices.extend([index1, index2, index3])

        # Create the cone base vertex
        position = np.array([0, 0, -half_height])[[x_dir, y_dir, z_dir]]
        normal = np.array([0, 0, -1])[[x_dir, y_dir, z_dir]]
        vertices.append([*position, *normal, 0.5, 0.5])

        # Create the cone base triangle fan
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x_dir, y_dir, z_dir = (
                (1, 2, 0),
                (2, 0, 1),
                (0, 1, 2)
            )[up_axis]

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            x = radius * cos_theta
            y = radius * sin_theta
            z = -half_height

            position = np.array([x, y, z])
            normal = np.array([0, 0, -1])
            uv = (cos_theta*0.5+0.5, sin_theta*0.5+0.5)

            vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
            vertices.append(vertex)
            
            index1 = i + segments + 1
            index2 = (i + 1) % segments + segments + 1
            index3 = len(vertices) - 1

            indices.extend([index1, index2, index3])

        vertex_data = np.array(vertices, dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)

        return vertex_data, index_data
    
    @staticmethod
    def _create_cylinder_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
        if up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be between 0 and 2")
        
        x_dir, y_dir, z_dir = (
            (1, 2, 0),
            (2, 0, 1),
            (0, 1, 2)
        )[up_axis]

        indices = []

        cap_vertices = []
        side_vertices = []

        # create center cap vertices
        position = np.array([0, 0, -half_height])[[x_dir, y_dir, z_dir]]
        normal = np.array([0, 0, -1])[[x_dir, y_dir, z_dir]]
        cap_vertices.append([*position, *normal, 0.5, 0.5])
        cap_vertices.append([*-position, *-normal, 0.5, 0.5])

        # Create the cylinder base and top vertices
        for j in (-1, 1):
            center_index = max(j, 0)
            for i in range(segments):
                theta = 2 * np.pi * i / segments

                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                x = radius * cos_theta
                y = radius * sin_theta
                z = j * half_height

                position = np.array([x, y, z])

                normal = np.array([x, y, 0])
                normal = normal / np.linalg.norm(normal)
                uv = (i / segments, (j + 1) / 2)
                vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
                side_vertices.append(vertex)

                normal = np.array([0, 0, j])
                uv = (cos_theta*0.5+0.5, sin_theta*0.5+0.5)
                vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
                cap_vertices.append(vertex)

                indices.extend([center_index, i+center_index*segments+2, (i+1)%segments+center_index*segments+2])

        # Create the cylinder side indices
        for i in range(segments):
            index1 = len(cap_vertices) + i + segments
            index2 = len(cap_vertices) + ((i + 1) % segments) + segments
            index3 = len(cap_vertices) + i + 1
            index4 = len(cap_vertices) + ((i + 1) % segments) + 1

            indices.extend([index1, index2, index3, index2, index4, index3])

        vertex_data = np.array(np.vstack((cap_vertices, side_vertices)), dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)

        return vertex_data, index_data
    
    @staticmethod
    def _create_box_mesh(extents):
        x_extent, y_extent, z_extent = extents
        half_x, half_y, half_z = x_extent / 2, y_extent / 2, z_extent / 2

        vertices = [
            # Position                  Normal    UV
            [-half_x, -half_y, -half_z, -1, 0, 0, 0, 0],
            [-half_x, -half_y,  half_z, -1, 0, 0, 1, 0],
            [-half_x,  half_y,  half_z, -1, 0, 0, 1, 1],
            [-half_x,  half_y, -half_z, -1, 0, 0, 0, 1],

            [half_x, -half_y, -half_z, 1, 0, 0, 0, 0],
            [half_x, -half_y,  half_z, 1, 0, 0, 1, 0],
            [half_x,  half_y,  half_z, 1, 0, 0, 1, 1],
            [half_x,  half_y, -half_z, 1, 0, 0, 0, 1],

            [-half_x, -half_y, -half_z, 0, -1, 0, 0, 0],
            [-half_x, -half_y,  half_z, 0, -1, 0, 1, 0],
            [ half_x, -half_y,  half_z, 0, -1, 0, 1, 1],
            [ half_x, -half_y, -half_z, 0, -1, 0, 0, 1],

            [-half_x,  half_y, -half_z, 0, 1, 0, 0, 0],
            [-half_x,  half_y,  half_z, 0, 1, 0, 1, 0],
            [ half_x,  half_y,  half_z, 0, 1, 0, 1, 1],
            [ half_x,  half_y, -half_z, 0, 1, 0, 0, 1],

            [-half_x, -half_y, -half_z, 0, 0, -1, 0, 0],
            [-half_x,  half_y, -half_z, 0, 0, -1, 1, 0],
            [ half_x,  half_y, -half_z, 0, 0, -1, 1, 1],
            [ half_x, -half_y, -half_z, 0, 0, -1, 0, 1],

            [-half_x, -half_y,  half_z, 0, 0, 1, 0, 0],
            [-half_x,  half_y,  half_z, 0, 0, 1, 1, 0],
            [ half_x,  half_y,  half_z, 0, 0, 1, 1, 1],
            [ half_x, -half_y,  half_z, 0, 0, 1, 0, 1],
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
