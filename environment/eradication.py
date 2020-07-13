import numpy as np
from .utils import grid_to_locs

def eradicate(actions, grid, beta_a = 3, beta_b = 4):
    action_locs = grid_to_locs(actions)
    killed = grid.astype(np.float32) * np.random.beta(beta_a, beta_b, size=grid.shape)
    mask = np.zeros_like(grid, dtype=np.float32)
    mask[action_locs[:, 0], action_locs[:, 1]] = 1.
    grid -= np.round(mask * killed).astype(np.int32)
    return grid


if __name__ == '__main__':
    # Unit test
    grid = np.random.randint(0, 10, (11, 11))
    action_locs = np.array([[1, 3], [2, 5], [6, 8], [9, 9]])
    print(grid)
    print(action_locs)
    print(eradicate(action_locs, grid))
