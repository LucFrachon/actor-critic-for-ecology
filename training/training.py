import torch
from environment.environment import InvasiveEnv
from agent.agent import Exerminator


def run_episode(agent, env, n_steps, ep_idx, initial_state = None):
    """
    :param initial_state: A (n_population, 2) np.ndarray with the initial positions of each population member.
        If not provided, starts with n_pop_ini in the centre of the grid.
    """
    step, score, is_terminal = 0, 0, False
    agent.current_state = torch.tensor(env.reset(initial_state), device=agent.device).reshape((
        1, 1, env.side_len, env.side_len))
    while (step < n_steps) and (not is_terminal):
        action = agent.pick_action()
        next_state, reward, is_terminal = env.step(action)
        # next_state = torch.tensor(next_state, device=device)
        # reward = torch.tensor(reward, device=device)
        agent.observe(next_state, reward, is_terminal)
        score += reward
        if is_terminal:
            break
        print(f"\rEpisode {ep_idx}, Step {step}: score {score}")
        step += 1
    return agent, score, step

def train(n_episodes,
          env_hparams, agent_hparams, actor_hparams, critic_hparams,
          n_steps=300, initial_state = None):
    env = InvasiveEnv(env_hparams, initial_state)
    agent = Exerminator(env.side_len, actor_hparams, critic_hparams, agent_hparams)
    episode_lens = []
    episode_rwds = []
    for e in range(n_episodes):
        agent, ep_score, ep_len = run_episode(agent, env, n_steps, e, initial_state)
        episode_lens.append(ep_len)
        episode_rwds.append(ep_score)

    return episode_lens, episode_rwds
