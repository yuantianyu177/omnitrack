import numpy as np

def gen_grid_np(h, w, normalize=False, homogeneous=False):
    if normalize:
        lin_y = np.linspace(-1., 1., num=h)
        lin_x = np.linspace(-1., 1., num=w)
    else:
        lin_y = np.arange(0, h)
        lin_x = np.arange(0, w)
    grid_x, grid_y = np.meshgrid(lin_x, lin_y)
    grid = np.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = np.concatenate([grid, np.ones_like(grid[..., :1])], axis=-1)
    return grid  # [h, w, 2 or 3]