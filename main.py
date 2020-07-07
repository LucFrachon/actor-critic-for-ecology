import numpy as np
from training.training import train
from settings import env_hparams, agent_hparams, actor_hparams, critic_hparams
from plotting.plotting import plot_episode_stats


if __name__ == '__main__':
    n_episodes = 100
    n_steps_per_ep = 300

    initial_state = np.random.randint(0, env_hparams['n_pop_ini'],
                                      size=(env_hparams['side_len'], env_hparams['side_len']))

    episode_lengths, episode_rewards, val_losses, pol_losses, action_locs = train(
        n_episodes,
        env_hparams, agent_hparams, actor_hparams, critic_hparams,
        n_steps_per_ep,
        initial_state=initial_state
    )
    plot_episode_stats(episode_lengths, episode_rewards, val_losses, pol_losses, action_locs, smoothing_window=5)
