import numpy as np


def locs_to_grid(locs, grid_dim):
    grid = np.zeros((grid_dim, grid_dim), dtype=np.int32)
    for x, y in locs:
        # Add a particle to the grid if it is actually on the grid!
        x = max(0, min(x, grid_dim - 1))
        y = max(0, min(y, grid_dim - 1))
        grid[x, y] += 1
    return grid

def grid_to_locs(grid):
    grid = grid.squeeze()
    non_zero_locs = np.nonzero(grid)
    n_particles = grid.sum(keepdims=False)
    locs = np.zeros((int(n_particles), 2), dtype=np.int32)
    offset = 0
    for i in range(len(non_zero_locs[0])):
        n_part_in_loc = int(grid[non_zero_locs[0][i], non_zero_locs[1][i]])
        for j in range(n_part_in_loc):
            locs[i + j + offset] = [non_zero_locs[0][i], non_zero_locs[1][i]]
        offset += j
    return locs