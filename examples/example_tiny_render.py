import warp as wp
import warp.render
import numpy as np

wp.init()

renderer = wp.render.TinyRenderer()

# renderer.render_capsule("capsule", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], radius=0.5, half_height=0.8)

time = 0.0
while renderer.is_running():
    renderer.begin_frame(time)
    for i in range(10):
        renderer.render_capsule(f"capsule_{i}", [0.2, np.sin(time), i-5.0], [0.0, 0.0, 0.0, 1.0], radius=0.5, half_height=0.8)
    renderer.render_cylinder("cylinder", [2.2, np.sin(time+0.5), 0.2], [0.0, 0.0, 0.0, 1.0], radius=0.5, half_height=0.8)
    renderer.render_cone("cone", [-1.2, np.sin(time+1.5), 0.0], np.array(wp.quat_identity()), radius=0.5, half_height=0.8)
    # renderer.render()
    renderer.end_frame()
    time += 1.0/60.0