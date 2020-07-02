import numpy as np


def reproduce(grid: np.ndarray, k: int) -> np.ndarray:
    """
    :param grid: numpy 2d-array representing the environment.
    :param k: ???
    # TODO: Vectorise this code
    """
    if grid.dtype == np.float32:
        grid = grid.astype(np.int32)
    mean_offspring = 1. / (1. + grid / k)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            for n in range(grid[r, c]):
                n_offspring = int(np.random.poisson(lam = mean_offspring[r, c]))
                grid[r, c] += n_offspring

    return grid


def death(grid: np.ndarray, death_rate: float) -> np.ndarray:
    """
    :param grid: numpy 2d-array representing the environment.
    :param death_rate: float
    """
    if grid.dtype == np.int32:
        grid = grid.astype(np.float32)
    grid *= 1 - death_rate
    return np.round(grid).astype(np.int32)


if __name__ == '__main__':
    # Unit test
    from environment.diffusion import diffuse
    from environment.utils import locs_to_grid

    nparticles = 100
    m = 11
    nitmax = 100

    locs = np.ones((nparticles, 2), dtype = np.int32) * m // 2
    grid = locs_to_grid(locs, m)
    print(grid)

    for _ in range(nitmax):
        _, grid = diffuse(grid, nparticles, .5)
        print("Diffusion:\n", grid)
        grid = reproduce(grid, 10)
        print("Reproduction:\n", grid)
        grid = death(grid, 0.1)
        print("Death:\n", grid)

