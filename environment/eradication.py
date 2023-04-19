import numpy as np
from utils import locs_to_grid


def eradicate(grid, actions, beta_a=3, beta_b=4, actions_as_grid=True):
    if not actions_as_grid:
        action_grid = locs_to_grid(actions, grid.shape[0])
    else:
        action_grid = actions
    killed = grid.astype(np.float32) * np.random.beta(beta_a, beta_b, size=grid.shape)
    killed = np.round(action_grid * killed).astype(np.int32)
    grid -= killed
    return grid


if __name__ == '__main__':
    # Unit test
    grid = np.random.randint(0, 10, (11, 11))
    action_locs = np.array([[1, 3], [2, 5], [6, 8], [9, 9]])
    print(action_locs, '\n')
    print(grid, '\n')
    print(eradicate(grid, action_locs, actions_as_grid=False), '\n')
