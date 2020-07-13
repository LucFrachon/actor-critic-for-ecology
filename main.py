import numpy as np
from training.training import train
from settings import env_hparams, agent_hparams, actor_hparams, critic_hparams
from plotting.plotting import plot_episode_stats


if __name__ == '__main__':
    # import cProfile

    n_episodes = 2
    n_steps_per_ep = 300

    # initial_state = np.random.randint(0, env_hparams['n_pop_ini'],
    #                                   size=(env_hparams['side_len'], env_hparams['side_len']))
    initial_state = None

    # cProfile.run('train(n_episodes,env_hparams, agent_hparams, actor_hparams, critic_hparams,n_steps_per_ep,'
    #              'initial_state=initial_state)', sort=1)
    episode_lengths, episode_rewards, val_losses, pol_losses, action_locs, pop_sizes, occupied_cells = train(
        n_episodes,
        env_hparams, agent_hparams, actor_hparams, critic_hparams,
        n_steps_per_ep,
        initial_state=initial_state
    )
    plot_episode_stats(episode_lengths, episode_rewards, val_losses, pol_losses, action_locs,
                       pop_sizes, occupied_cells, smoothing_window=5, show=True, save=True,
                       save_dir='./ep10_steps300_count_noinitialstate')
