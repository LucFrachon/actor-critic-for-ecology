import numpy as np


def locs_to_grid(locs, grid_dim):
    unique, counts = np.unique(locs, axis=0, return_counts=True)
    unique = np.clip(unique, 0, grid_dim - 1)
    grid = np.zeros((grid_dim, grid_dim), dtype=int)
    grid[unique[:, 0], unique[:, 1]] = counts
    return grid


def grid_to_locs(grid):
    grid = grid.squeeze().astype(int)
    x, y = np.nonzero(grid)
    return np.c_[x, y].repeat(grid[x, y], axis=0)
