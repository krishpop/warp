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

void main()
{
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(vec3(-0.2, 1.0, 0.3));
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    // checkerboard pattern
    float u = TexCoord.x;
    float v = TexCoord.y;
    float scale = 2.0;
    float checker = mod(floor(u * scale) + floor(v * scale), 2.0);
    vec3 checkerColor = mix(ObjectColor1, ObjectColor2, checker);

    vec3 result = (ambient + diffuse + specular) * checkerColor;
    FragColor = vec4(result, 1.0);
}
'''

grid_vertex_shader = '''
#version 330

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
#version 330

out vec4 outColor;

void main() {
    outColor = vec4(1.0);
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
    scaling: float,
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44)):

    tid = wp.tid()
    i = instance_id[tid]
    body = instance_body[i]
    X_ws = instance_transforms[i]
    if body >= 0:
        if body_q:
            X_ws = body_q[body] * X_ws
        else:
            return
    p = wp.transform_get_translation(X_ws)
    q = wp.transform_get_rotation(X_ws)
    p *= scaling
    s = instance_scalings[i] * scaling
    rot = wp.quat_to_matrix(q)
    # transposed definition
    vbo_transforms[tid] = wp.mat44(
        rot[0,0]*s[0], rot[1,0]*s[1], rot[2,0]*s[2], 0.0,
        rot[0,1]*s[0], rot[1,1]*s[1], rot[2,1]*s[2], 0.0,
        rot[0,2]*s[0], rot[1,2]*s[1], rot[2,2]*s[2], 0.0,
        p[0], p[1], p[2], 1.0)


@wp.kernel
def update_points_positions(
    instance_id: wp.array(dtype=int),
    position: wp.array(dtype=wp.vec3),
    scaling: float,
    # outputs
    vbo_positions: wp.array(dtype=wp.vec4)):

    tid = wp.tid()
    p = position[tid] * scaling
    vbo_positions[instance_id[tid]] = wp.vec4(p[0], p[1], p[2], 0.0)


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


def check_gl_error():
    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        print(f"OpenGL error: {error}")


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
        near_plane=0.001,
        far_plane=1000.0,
        camera_fov=45.0,
        background_color=(0.5, 0.5, 0.5),
        draw_grid=True,
    ):
        
        self.scaling = scaling
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.camera_fov = camera_fov
        self.background_color = background_color
        self.draw_grid = draw_grid

        self._device = wp.get_cuda_device()

        if not glfw.init():
            raise Exception("GLFW initialization failed!")
        
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)

        self.window = glfw.create_window(screen_width, screen_height, title, None, None)

        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed!")
        
        self._camera_pos = glm.vec3(0.0, 2.0, 10.0)
        self._camera_front = glm.vec3(0.0, 0.0, -1.0)
        self._camera_up = glm.vec3(0.0, 1.0, 0.0)
        self._camera_speed = 0.1
        self._camera_axis = "xyz".index(upaxis.lower())
        self._yaw, self._pitch = -90.0, 0.0
        self._last_x, self._last_y = 800 // 2, 600 // 2
        self._first_mouse = True
        self._left_mouse_pressed = False
        self._keys_pressed = defaultdict(bool)

        self.time = 0.0
        self.paused = False
        self.skip_rendering = False
        self._skip_frame_counter = 0

        self._body_name = {}
        self._shapes = []
        self._shape_geo_hash = {}
        self._shape_gl_buffers = {}
        self._shape_instances = defaultdict(list)
        self._instances = {}
        self._instance_gl_buffers = {}
        self._instance_transform_gl_buffer = None
        self._instance_transform_cuda_buffer = None
        self._instance_color1_buffer = None
        self._instance_color2_buffer = None
        self._instance_count = 0
        self._wp_instance_ids = None
        self._wp_instance_transforms = None
        self._wp_instance_scalings = None
        self._wp_instance_bodies = None
        self._update_shape_instances = False
        self._add_shape_instances = False
        

        glfw.make_context_current(self.window)

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
        
        

        width, height = glfw.get_window_size(self.window)
        self._projection_matrix = glm.perspective(np.deg2rad(45), width / height, self.near_plane, self.far_plane)
        # glUniformMatrix4fv(self._loc_shape_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))

        self._view_matrix = glm.lookAt(self._camera_pos, self._camera_pos + self._camera_front, self._camera_up)
        # glUniformMatrix4fv(self._loc_shape_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        
        # if self._camera_axis == 0:
        #     self._model_matrix = glm.mat4(self.scaling, 0, 0, 0,
        #                      0, 0, self.scaling, 0,
        #                      0, -self.scaling, 0, 0,
        #                      0, 0, 0, 1)
        # elif self._camera_axis == 2:
        #     self._model_matrix = glm.mat4(0, 0, -self.scaling, 0,
        #                      0, self.scaling, 0, 0,
        #                      self.scaling, 0, 0, 0,
        #                      0, 0, 0, 1)
        # else:
        #     self._model_matrix = glm.mat4(self.scaling, 0, 0, 0,
        #                      0, self.scaling, 0, 0,
        #                      0, 0, self.scaling, 0,
        #                      0, 0, 0, 1)
        self._model_matrix = glm.mat4(1.0)
        # glUniformMatrix4fv(self._loc_shape_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))

        glUseProgram(self._grid_shader)

        # create grid data
        limit = 10.0
        ticks = np.linspace(-limit, limit, 21)
        grid_vertices = []
        for i in ticks:
            grid_vertices.extend([-limit, 0, i, limit, 0, i])
            grid_vertices.extend([i, 0, -limit, i, 0, limit])
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
        
        glBindVertexArray(0)
        
        glUniformMatrix4fv(self._loc_grid_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))
        glUniformMatrix4fv(self._loc_grid_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        glUniformMatrix4fv(self._loc_grid_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))

        import pycuda.gl.autoinit





        if False:
            glUseProgram(self._shape_shader)

            glUniformMatrix4fv(self._loc_shape_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
            glUniform3f(self._loc_shape_view_pos, *self._camera_pos)
            glUniformMatrix4fv(self._loc_shape_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
            glUniformMatrix4fv(self._loc_shape_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))
            glUniformMatrix4fv(self._loc_shape_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))

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
            
            glUseProgram(self._shape_shader)

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
            glUseProgram(self._shape_shader)

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


            instance_buffer_cuda = pycuda.gl.RegisteredBuffer(int(instance_buffer))
            mapped_buffer = instance_buffer_cuda.map()
            ptr, _ = mapped_buffer.device_ptr_and_size()
            mapped_buffer.unmap()

            wp_positions = wp.array(instance_positions, dtype=wp.vec3, device=self._device)
            wp_scalings = wp.array(np.random.uniform(0.5, 2.0, size=(num_instances, 3)), dtype=wp.vec3, device=self._device)

            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self._process_input(self.window)
                glClearColor(*self.background_color, 1)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                
                view = glm.lookAt(self._camera_pos, self._camera_pos + self._camera_front, self._camera_up)
                
                self._view_matrix = glm.lookAt(self._camera_pos, self._camera_pos + self._camera_front, self._camera_up)

                imgui.new_frame()
                imgui.set_next_window_bg_alpha(0.0)
                imgui.set_next_window_position(0, 0)
                imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)
                imgui.begin("Custom window", True,
                            imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR)
                # imgui.text(f"FPS: {1.0 / duration:.1f}")
                imgui.spacing()
                imgui.text(f"Shapes: {len(self._shapes)}")
                imgui.text(f"Instances: {len(self._instances)}")
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

                # draw grid
                if self.draw_grid:
                    
                    glUseProgram(self._grid_shader)
                    
                    glUniformMatrix4fv(self._loc_grid_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
                    glUniformMatrix4fv(self._loc_grid_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))
                    glUniformMatrix4fv(self._loc_grid_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))

                    glBindVertexArray(self._grid_vao)

                    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

                    # glEnableVertexAttribArray(0)
                    # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._grid_vbo)
                    # glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0)

                    # # Enable vertex array and set data
                    # # glEnableClientState(GL_VERTEX_ARRAY)
                    # # glVertexPointer(3, GL_FLOAT, 0, None)
                    
                    # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._grid_ebo)

                    # Draw lines
                    glDrawArrays(GL_LINES, 0, self._grid_vertex_count)
                    # glDrawArrays(GL_TRIANGLES, 0, self._grid_vertex_count)
                    # glDrawElements(GL_LINES, self._grid_vertex_count, GL_UNSIGNED_INT, None)

                    
                    # Disable vertex array and unbind VBO
                    # glDisableClientState(GL_VERTEX_ARRAY)
                    # glBindBuffer(GL_ARRAY_BUFFER, 0)
                    # glDisableVertexAttribArray(0)

                glUseProgram(self._shape_shader)
                glUniformMatrix4fv(self._loc_shape_view, 1, GL_FALSE, glm.value_ptr(view))

                glUniform3f(self._loc_shape_view_pos, *self._camera_pos)
                
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
                    ptr, _ = mapped_buffer.device_ptr_and_size()
                    wp_instance_buffer = wp.array(dtype=wp.mat44, shape=(num_instances,), device=self._device, ptr=ptr, owner=False)
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

                self.imgui_renderer.render(imgui.get_draw_data())

                glfw.swap_buffers(self.window)



        # # Clean up
        # glDeleteVertexArrays(1, [vao])
        # glDeleteBuffers(1, [vbo])
        # glDeleteBuffers(1, [ebo])
        # glDeleteBuffers(1, [instance_buffer])
        # glfw.terminate()

        self._last_time = glfw.get_time()

    def clear(self):
        for vao, vbo, ebo, _ in self._shape_gl_buffers.values():
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
            glDeleteBuffers(1, [ebo])
        if self._instance_transform_gl_buffer is not None:
            glDeleteBuffers(1, [self._instance_transform_gl_buffer])
            glDeleteBuffers(1, [self._instance_color1_buffer])
            glDeleteBuffers(1, [self._instance_color2_buffer])
        
        self._body_name.clear()
        self._shapes.clear()
        self._shape_geo_hash.clear()
        self._shape_gl_buffers.clear()
        self._shape_instances.clear()
        self._instances.clear()
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

    def update_projection_matrix(self):
        resolution = glfw.get_framebuffer_size(self.window)
        if resolution[1] == 0:
            return
        aspect_ratio = resolution[0] / resolution[1]
        self._projection_matrix = glm.perspective(glm.radians(self.camera_fov), aspect_ratio, self.near_plane, self.far_plane)
    
    def begin_frame(self, time: float):
        self.time = time

    def end_frame(self):
        if self._add_shape_instances:
            self.add_shape_instances()
        if self._update_shape_instances:
            self.update_shape_instances()
        self.update()
        while self.paused and self.is_running():
            self.update()

    def update(self):
        self._skip_frame_counter += 1
        if self._skip_frame_counter > 100:
            self._skip_frame_counter = 0
        if self.skip_rendering:
            if self._skip_frame_counter == 0:
                # ensure we receive key events
                glfw.poll_events()
                self._process_input(self.window)
            return
        
        duration = glfw.get_time() - self._last_time
        self._last_time = glfw.get_time()
        
        glfw.poll_events()
        self.imgui_renderer.process_inputs()
        self._process_input(self.window)
        
        glClearColor(*self.background_color, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindVertexArray(0)

        self._view_matrix = glm.lookAt(self._camera_pos, self._camera_pos + self._camera_front, self._camera_up)

        imgui.new_frame()
        imgui.set_next_window_bg_alpha(0.0)
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(*glfw.get_framebuffer_size(self.window))
        imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)
        imgui.begin("Custom window", True,
                    imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR)
        imgui.text(f"FPS: {1.0 / duration:.1f}")
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

        # draw grid
        if self.draw_grid:
            
            glUseProgram(self._grid_shader)
            
            glUniformMatrix4fv(self._loc_grid_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
            glUniformMatrix4fv(self._loc_grid_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))
            glUniformMatrix4fv(self._loc_grid_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))

            glBindVertexArray(self._grid_vao)
            glDrawArrays(GL_LINES, 0, self._grid_vertex_count)
            glBindVertexArray(0)

        
        glUseProgram(self._shape_shader)
        glUniformMatrix4fv(self._loc_shape_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        glUniform3f(self._loc_shape_view_pos, *self._camera_pos)
        glUniformMatrix4fv(self._loc_shape_view, 1, GL_FALSE, glm.value_ptr(self._view_matrix))
        glUniformMatrix4fv(self._loc_shape_projection, 1, GL_FALSE, glm.value_ptr(self._projection_matrix))
        glUniformMatrix4fv(self._loc_shape_model, 1, GL_FALSE, glm.value_ptr(self._model_matrix))

        self._render_scene()
        
        # Check for OpenGL errors
        check_gl_error()

        self.imgui_renderer.render(imgui.get_draw_data())

        glfw.swap_buffers(self.window)

    def _render_scene(self):
        start_instance_idx = 0
        for shape, (vao, vbo, ebo, tri_count) in self._shape_gl_buffers.items():
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

            num_instances = len(self._shape_instances[shape])
            # glDrawElementsInstanced(GL_TRIANGLES, tri_count, GL_UNSIGNED_INT, None, num_instances)
            glDrawElementsInstancedBaseInstance(GL_TRIANGLES, tri_count, GL_UNSIGNED_INT, None, num_instances, start_instance_idx)
            # glDrawArrays(GL_TRIANGLES, 0, tri_count)

            start_instance_idx += num_instances

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
            self._camera_pos += self._camera_speed * self._camera_front
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS or glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            self._camera_pos -= self._camera_speed * self._camera_front
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS or glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            self._camera_pos -= self._camera_speed * glm.normalize(glm.cross(self._camera_front, self._camera_up))
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            self._camera_pos += self._camera_speed * glm.normalize(glm.cross(self._camera_front, self._camera_up))
        
        if self._pressed_key(glfw.KEY_ESCAPE):
            glfw.set_window_should_close(window, True)
        if self._pressed_key(glfw.KEY_SPACE):
            self.paused = not self.paused
        if self._pressed_key(glfw.KEY_TAB):
            self.skip_rendering = not self.skip_rendering
    
    def _scroll_callback(self, window, x_offset, y_offset):
        self.camera_fov -= y_offset
        self.camera_fov = max(min(self.camera_fov, 90.0), 15.0)
        self.update_projection_matrix()

    def _window_resize_callback(self, window, width, height):
        self._first_mouse = True
        glViewport(0, 0, width, height)
        self.update_projection_matrix()
    
    def register_shape(self, geo_hash, vertices, indices, color1=None, color2=None):
        shape = len(self._shapes)
        if color1 is None:
            color1 = tab10_color_map(len(self._shape_geo_hash))
        if color2 is None:
            color2 = np.clip(np.array(color1) + 0.25, 0.0, 1.0)
        # TODO check if we actually need to store the shape data
        self._shapes.append((vertices, indices, color1, color2))
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
            color2 = np.clip(np.array(color1) + 0.25, 0.0, 1.0)
        instance = len(self._instances)
        self._shape_instances[shape].append(instance)
        body = self._resolve_body_id(body)
        self._instances[name] = (instance, body, shape, [*pos, *rot], scale, color1, color2)
        self._add_shape_instances = True
        self._instance_count = len(self._instances)
        return instance
    
    def add_shape_instances(self):
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

        # Create CUDA buffer
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
        
        # trigger update to the instance transforms
        self._update_shape_instances = True

        self._wp_instance_ids = wp.array(instance_ids, dtype=wp.int32, device=self._device)

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
                self.scaling,
            ],
            outputs=[
                vbo_transforms,
            ],
            device="cuda")
        
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

    def render_plane(self, name: str, pos: tuple, rot: tuple, width: float, length: float, color: tuple=(1.,1.,1.), texture=None, parent_body: str=None, is_template: bool=False, u_scaling=1.0, v_scaling=1.0):
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
            faces = np.array([0, 1, 2, 2, 3, 0])
            normal = (0.0, 1.0, 0.0)
            width = (width if width > 0.0 else 100.0) * self.scaling
            length = (length if length > 0.0 else 100.0) * self.scaling
            aspect = width / length
            u = width * aspect * u_scaling / self.scaling
            v = length * v_scaling / self.scaling
            gfx_vertices = np.array([
                [-width, 0.0, -length, 0.0, *normal, 0.0, 0.0],
                [-width, 0.0,  length, 0.0, *normal, 0.0, v],
                [width, 0.0,  length, 0.0, *normal, u, v],
                [width, 0.0, -length, 0.0, *normal, u, 0.0],
            ])
            shape = self.register_shape(geo_hash, gfx_vertices, faces, color1=color)
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
        return self.render_plane("ground", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), size, size, color2, u_scaling=1.0, v_scaling=1.0)
    
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
        if up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be between 0 and 2")

        vertices = []
        indices = []

        h = 2*half_height
        cone_angle = np.arctan2(radius, h)
        cos_angle = np.cos(cone_angle)
        sin_angle = np.sin(cone_angle)

        x_dir, y_dir, z_dir = (
            (1, 2, 0),
            (0, 1, 2),
            (2, 0, 1),
        )[up_axis]

        # Create the cone side vertices
        for i in range(segments):
            theta = 2 * np.pi * i / segments

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            x = radius * cos_theta
            y = -half_height
            z = radius * sin_theta

            position = np.array([x, y, z])
            normal = np.array([cos_angle*cos_theta, sin_angle, cos_angle*sin_theta])
            uv = (cos_theta*0.5 + 0.5, 0.0)

            vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
            vertices.append(vertex)

        # Create the cone tip vertex
        position = np.array([0, half_height, 0])[[x_dir, y_dir, z_dir]]
        normal = np.array([0, 1, 0])[[x_dir, y_dir, z_dir]]
        vertices.append([*position, *normal, 0.5, 1])

        # Create the cone side indices
        for i in range(segments):
            index1 = i
            index2 = (i + 1) % segments
            index3 = segments
            indices.extend([index1, index2, index3])

        # Create the cone base vertex
        position = np.array([0, -half_height, 0])[[x_dir, y_dir, z_dir]]
        normal = np.array([0, -1, 0])[[x_dir, y_dir, z_dir]]
        vertices.append([*position, *normal, 0.5, 0.5])

        # Create the cone base triangle fan
        for i in range(segments):
            theta = 2 * np.pi * i / segments

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            x = radius * cos_theta
            y = -half_height
            z = radius * sin_theta

            position = np.array([x, y, z])
            normal = np.array([0, -1, 0])
            uv = (cos_theta*0.5+0.5, sin_theta*0.5+0.5)

            vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
            vertices.append(vertex)
            
            index1 = i + segments + 2
            index2 = (i + 1) % segments + segments + 2
            index3 = segments + 1

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

        # Create the cylinder base and top vertices
        for j in (-1, 1):
            center_index = max(j, 0)
            for i in range(segments):
                theta = 2 * np.pi * i / segments

                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                x = radius * cos_theta
                y = j * half_height
                z = radius * sin_theta

                position = np.array([x, y, z])

                normal = np.array([x, 0, z])
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
