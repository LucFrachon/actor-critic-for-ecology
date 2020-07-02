import numpy as np
from environment.utils import locs_to_grid, grid_to_locs


def diffuse(grid, sigma_diff=1.):
    locs = grid_to_locs(grid)
    # Update the particles' locations at random.
    mvmt = np.round(sigma_diff * np.random.standard_normal(locs.shape)).astype('int')
    locs += mvmt
    # Create an updated grid and plot it.
    grid = locs_to_grid(locs, grid.shape[0])
    return grid


if __name__ == "__main__":
    # Unit test
    # grid side length
    m = 11
    # Maximum numbter of iterations.
    nitmax = 1000
    # Number of particles in the simulation.
    nparticles = 1000


    locs = np.ones((nparticles, 2), dtype=int) * m//2
    grid = locs_to_grid(locs, m)
    print(grid)
    for _ in range(nitmax):
        locs, grid = diffuse(grid, nparticles, .5)
        print(grid)
    print(locs)
