from training.training import train
from settings import env_hparams, agent_hparams, actor_hparams, critic_hparams
from plotting.plotting import plot_episode_stats


if __name__ == '__main__':
    n_episodes = 300
    n_steps_per_ep = 300

    episode_lengths, episode_rewards = train(
        n_episodes,
        env_hparams, agent_hparams, actor_hparams, critic_hparams,
        n_steps_per_ep,
        initial_state=None
    )
    plot_episode_stats()


