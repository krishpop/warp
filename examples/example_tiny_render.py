import warp as wp
import warp.render
import numpy as np
import matplotlib.pyplot as plt

wp.init()

renderer = wp.render.TinyRenderer()
renderer.setup_tiled_rendering([np.arange(13)]*10)

renderer.render_ground()

pixels = wp.zeros((renderer.screen_height, renderer.screen_width, 3), dtype=wp.float32)
plt.figure(1)
img_plot = plt.imshow(pixels.numpy())
plt.ion()
plt.show()

while renderer.is_running():
    time = renderer.clock_time
    renderer.begin_frame(time)
    for i in range(10):
        renderer.render_capsule(f"capsule_{i}", [1.2, np.sin(time+i*0.2), i-5.0], [0.0, 0.0, 0.0, 1.0], radius=0.5, half_height=0.8)
    renderer.render_cylinder("cylinder", [3.2, 1.0, np.sin(time+0.5)], np.array(wp.quat_from_axis_angle((1.0, 0.0, 0.0), np.sin(time+0.5))), radius=0.5, half_height=0.8)
    renderer.render_cone("cone", [-1.2, 1.0, 0.0], np.array(wp.quat_from_axis_angle((0.707, 0.707, 0.0), time)), radius=0.5, half_height=0.8)
    renderer.end_frame()

    if plt.fignum_exists(1):
        if (renderer.screen_height, renderer.screen_width, 3) != pixels.shape:
            # make sure we resize the pixels array to the right dimensions if the user resizes the window
            pixels = wp.zeros((renderer.screen_height, renderer.screen_width, 3), dtype=wp.float32)

        renderer.get_pixels(pixels)
        img_plot.set_data(pixels.numpy())
renderer.clear()
