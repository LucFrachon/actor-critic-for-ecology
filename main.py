import os
import numpy as np
from training.training import train
from settings import env_hparams, agent_hparams, actor_hparams, critic_hparams
from plotting.plotting import plot_episode_stats


if __name__ == '__main__':

    n_episodes = 200
    n_steps_per_ep = 1000
    save_plots = True
    if save_plots:
        save_dir = f'./plots/ep{n_episodes}_' \
                   f'steps{n_steps_per_ep}_' \
                   f'reward-{env_hparams["reward_method"]}_' \
                   f'env-size{env_hparams["side_len"]}'
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    # initial_state = None
    # sample a binary mask from a Bernoulli distribution with probability env_hparams['proportion_occupied']
    occupancy_mask = np.random.binomial(
        1,
        env_hparams['proportion_occupied'],
        size=(env_hparams['side_len'], env_hparams['side_len'])
    )
    initial_state = occupancy_mask * np.random.randint(
        0,
        env_hparams['n_pop_ini'],
        size=(env_hparams['side_len'], env_hparams['side_len'])
    )
    print(f"Initial state: {occupancy_mask.sum()} occupied cells, {initial_state.sum()} total population:")
    print(initial_state)

    episode_lengths, episode_rewards, val_losses, pol_losses, action_locs, pop_sizes, occupied_cells = train(
        n_episodes,
        env_hparams,
        agent_hparams,
        actor_hparams,
        critic_hparams,
        n_steps_per_ep,
        initial_state=initial_state,
        print_every=1,
    )
    plot_episode_stats(
        episode_lengths,
        episode_rewards,
        val_losses,
        pol_losses,
        action_locs,
        pop_sizes,
        occupied_cells,
        smoothing_window=5,
        show=False,
        save=save_plots,
        save_dir=save_dir
    )
