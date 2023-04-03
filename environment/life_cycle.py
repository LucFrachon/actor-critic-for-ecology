from copy import copy
import numpy as np


def reproduce(grid: np.ndarray, k: int, min_to_reproduce: int = 2) -> np.ndarray:
    """
    :param grid: numpy 2d-array representing the environment.
    :param k: ???
    :param min_to_reproduce: minimum number of individuals in a cell for the organism to be able to reproduce.
    """
    grid = copy(grid)
    grid = grid.astype(np.int32)
    mean_offspring = (grid >= min_to_reproduce) * (1 + 1. / (1. + grid / k))  # TODO: this is monotonically decreasing in x, is it correct?
    n_offspring_per_ind = np.random.poisson(lam=mean_offspring)
    # The number of individuals added to the population is the sum of the number of offspring of each set of parents.
    # Therefore, if two parents are required, we should divide the number of offspring by 2
    # TODO: Check this logic with Alice
    # print(f"{n_offspring_per_ind = }")
    grid = n_offspring_per_ind * grid // min_to_reproduce + grid
    return grid


def death(grid: np.ndarray, death_rate: float) -> np.ndarray:
    """
    :param grid: numpy 2d-array representing the environment.
    :param death_rate: float
    """
    grid = copy(grid)
    # grid = grid.astype(np.float32)
    # grid *= (1 - death_rate)  # TODO: shouldn't this have some stochasticity and depend on the number of individuals?

    # Proposed solution:
    # This function behaves like a constant for small values of x, and like an exponential for large values.
    true_death_rate = death_rate + np.exp(death_rate * grid / 10.) - np.exp(death_rate * (grid - 1) / 10.)
    n_deaths = np.random.binomial(n=grid, p=np.clip(true_death_rate, death_rate, 1. - death_rate))
    return np.round(grid - n_deaths).astype(np.int32)


if __name__ == '__main__':
    # Unit test
    from diffusion import diffuse
    from utils import locs_to_grid

    n_particles = 100
    grid_size = 11
    n_iterations = 10

    locs = np.ones((n_particles, 2), dtype = np.int32) * grid_size // 2
    grid = locs_to_grid(locs, grid_size)
    print(grid)

    for _ in range(n_iterations):
        grid = diffuse(grid, sigma_diff=0.25, clamp=True)
        print("Diffusion:\n", grid)
        grid = reproduce(grid, k=10, min_to_reproduce=2)
        print("Reproduction:\n", grid)
        grid = death(grid, 0.1)
        print("Death:\n", grid)
