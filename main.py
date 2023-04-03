import os
import numpy as np
from training.training import train
from settings import env_hparams, agent_hparams, actor_hparams, critic_hparams
from plotting.plotting import plot_episode_stats


if __name__ == '__main__':
    # import cProfile


    n_episodes = 30
    n_steps_per_ep = 100
    save_plots = True
    if save_plots:
        save_dir = './ep100_steps100_sum_51x51'
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    # initial_state = None
    initial_state = np.random.randint(0, env_hparams['n_pop_ini'],
                                      size=(env_hparams['side_len'], env_hparams['side_len']))

    # cProfile.run('train(n_episodes,env_hparams, agent_hparams, actor_hparams, critic_hparams,n_steps_per_ep,'
    #              'initial_state=initial_state)', sort=1)
    episode_lengths, episode_rewards, val_losses, pol_losses, action_locs, pop_sizes, occupied_cells = train(
        n_episodes,
        env_hparams,
        agent_hparams,
        actor_hparams,
        critic_hparams,
        n_steps_per_ep,
        initial_state=initial_state
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
        show=True,
        save=save_plots,
        save_dir=save_dir
    )
