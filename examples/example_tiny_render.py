import warp as wp
import warp.render
import numpy as np
import matplotlib.pyplot as plt

wp.init()

# number of viewports to render in a single frame
num_tiles = 9
# whether to split tiles into subplots
split_up_tiles = True

renderer = wp.render.TinyRenderer()
instance_ids = []
# set up instances to hide one of the capsules in each tile
for i in range(num_tiles):
    instances = [j for j in np.arange(13) if j != i+2]
    instance_ids.append(instances)
renderer.setup_tiled_rendering(instance_ids)

renderer.render_ground()


if split_up_tiles:
    pixels = wp.zeros((num_tiles, renderer.tile_height, renderer.tile_width, 3), dtype=wp.float32)
    ncols = int(np.ceil(np.sqrt(num_tiles)))
    nrows = int(np.ceil(num_tiles / float(ncols)))
    img_plots = []
    aspect_ratio = renderer.tile_height / renderer.tile_width
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        constrained_layout=True,
        figsize=(ncols * 3.5, nrows * 3.5 * aspect_ratio),
        squeeze=False,
        sharex=True,
        sharey=True,
        num=1
    )
    tile_temp = np.zeros((renderer.tile_height, renderer.tile_width, 3), dtype=np.float32)
    for dim in range(ncols * nrows):
        ax = axes[dim // ncols, dim % ncols]
        if dim >= num_tiles:
            ax.axis("off")
            continue
        img_plots.append(ax.imshow(tile_temp))
else:
    plt.figure(1)
    pixels = wp.zeros((renderer.screen_height, renderer.screen_width, 3), dtype=wp.float32)
    img_plot = plt.imshow(pixels.numpy())
plt.ion()
plt.show()

while renderer.is_running():
    time = renderer.clock_time
    renderer.begin_frame(time)
    for i in range(10):
        renderer.render_capsule(f"capsule_{i}", [i-5.0, np.sin(time+i*0.2), -3.0], [0.0, 0.0, 0.0, 1.0], radius=0.5, half_height=0.8)
    renderer.render_cylinder("cylinder", [3.2, 1.0, np.sin(time+0.5)], np.array(wp.quat_from_axis_angle((1.0, 0.0, 0.0), np.sin(time+0.5))), radius=0.5, half_height=0.8)
    renderer.render_cone("cone", [-1.2, 1.0, 0.0], np.array(wp.quat_from_axis_angle((0.707, 0.707, 0.0), time)), radius=0.5, half_height=0.8)
    renderer.end_frame()

    if plt.fignum_exists(1):
        if split_up_tiles:
            pixel_shape = (num_tiles, renderer.tile_height, renderer.tile_width, 3)
        else:
            pixel_shape = (renderer.screen_height, renderer.screen_width, 3)

        if pixel_shape != pixels.shape:
            # make sure we resize the pixels array to the right dimensions if the user resizes the window
            pixels = wp.zeros(pixel_shape, dtype=wp.float32)

        renderer.get_pixels(pixels, split_up_tiles=split_up_tiles)

        if split_up_tiles:
            pixels_np = pixels.numpy()
            for i, img_plot in enumerate(img_plots):
                img_plot.set_data(pixels_np[i])
        else:
            img_plot.set_data(pixels.numpy())
        fig.canvas.draw()
        fig.canvas.flush_events()

renderer.clear()
