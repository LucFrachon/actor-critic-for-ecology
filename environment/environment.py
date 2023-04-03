from argparse import Namespace
import numpy as np
from typing import Optional, Tuple
from environment.utils import locs_to_grid
from environment.life_cycle import reproduce, death
from environment.diffusion import diffuse
from environment.eradication import eradicate

np.set_printoptions(linewidth=320)


class InvasiveEnv:
    def __init__(self, hparams, initial_state=None):
        """
        :param initial_state: A (n_population, 2) np.ndarray with the initial positions of each population member.
        If not provided, starts with self.n_pop_ini in the centre of the grid.
        """
        hparams = Namespace(**hparams)
        self.side_len = hparams.side_len
        # We'll leave this aside for now and assume actions are binary:
        # self.action_space = action_space  # this needs to be a np.ndarray of discrete action strengths
        self.death_rate = hparams.death_rate
        self.disp_sigma = hparams.disp_sigma
        self.erad_alpha, self.erad_beta = hparams.erad_alpha, hparams.erad_beta
        self.k = hparams.k  # used in reproduce
        self.mgmt_cost = hparams.mgmt_cost
        self.eradication_bonus = hparams.eradication_bonus
        self.n_pop_ini = hparams.n_pop_ini
        # Rewards can either be computed by summing up the whole remaining population, or by counting the
        # number of infected cells on the grid. Other methods can be implemented in self.compute_reward.
        assert hparams.reward_method in ['sum', 'count']
        self.reward_method = hparams.reward_method
        self.normalise_reward = hparams.normalise_reward
        self.normalise_cost = hparams.normalise_cost
        self.grid = None

    def reset(self, initial_state: Optional[np.ndarray]) -> np.ndarray:
        """
        initial_state: Same shape as env grid. Initial population in each cell of the env.
        If none is passed, all the population is stacked on the centre cell.
        :return New grid layout, as a Numpy 2d-array of integers
        """
        if initial_state is None:

            initial_locs = np.ones((self.n_pop_ini, 2), dtype=int) * self.side_len // 2
            self.grid = locs_to_grid(initial_locs, self.side_len)
        else:
            self.grid = initial_state
        return self.grid.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        :param action: Grid with 0s and 1s depending on whether the agent acts on each location.
        Has same shape as environment grid.
        :return next_state, reward, done: is the environment in its terminal state?
        """
        # Dispersion
        self.grid = diffuse(self.grid, self.disp_sigma)
        # Reproduce
        self.grid = reproduce(self.grid, self.k)
        # Die
        self.grid = death(self.grid, self.death_rate)
        # Eradication
        self.grid = eradicate(self.grid, action, self.erad_alpha, self.erad_beta)
        # Management cost
        cost = action.sum() * self.mgmt_cost
        # Is this the terminal state?
        done = self.grid.sum() == 0
        if done:
            bonus = self.eradication_bonus
        else:
            bonus = 0
        # Compute reward
        reward = self.compute_reward(cost) + bonus
        return self.grid.astype(np.float32), reward, done

    def compute_reward(self, cost):
        if self.reward_method == 'sum':
            reward = -np.sum(self.grid).astype(np.float32)
        else:  # 'count'
            reward = -float(np.count_nonzero(self.grid))
        if self.normalise_reward:
            reward /= (self.side_len * self.side_len)
        if self.normalise_cost:
            cost /= (self.side_len * self.side_len)

        return reward - cost

    def __repr__(self):
        if self.grid:
            return f'Environment with {self.grid.sum()} individuals:\n{self.grid}'
        else:
            return 'Undefined environment. Run .reset() first.'


if __name__ == '__main__':
    # Unit test
    env = InvasiveEnv(side_len=5)
    print(env.grid)
    for _ in range(100):
        action_locs = np.random.randint(0, 5, size=(4, 2))
        print(action_locs)
        env.step(action_locs)
        print(env.grid)

"""
hparam:
        side_len
        death_rate
        disp_sigma
        erad_alpha
        erad_beta
        k
        mgmt_cost
        eradication_bonus
        n_pop_ini
        reward_method
"""
