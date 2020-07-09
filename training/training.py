import torch
from environment.environment import InvasiveEnv
from agent.agent import Exerminator


def run_episode(agent, env, n_steps, ep_idx, initial_state = None):
    """
    :param initial_state: A (n_population, 2) np.ndarray with the initial positions of each population member.
        If not provided, starts with n_pop_ini in the centre of the grid.
    """
    step, score, is_terminal = 0, 0, False
    val_losses, pol_losses = [], []
    action_locs = []
    agent.current_state = torch.tensor(env.reset(initial_state), device=agent.device).reshape((
        1, 1, env.side_len, env.side_len))
    if agent.hparams.normalise_states:
        agent.current_state = agent.normalise_state(agent.current_state)  # normalise
    while (step < n_steps) and (not is_terminal):
        action = agent.pick_action()
        action_locs.append(action.sum())
        next_state, reward, is_terminal = env.step(action)
        # next_state = torch.tensor(next_state, device=device)
        # reward = torch.tensor(reward, device=device)
        agent.observe(next_state, reward, is_terminal)
        val_loss, pol_loss = agent.train()
        val_losses.append(val_loss)
        pol_losses.append(pol_loss)
        score += reward
        if is_terminal:
            break
        print(f"\rEpisode {ep_idx}, Step {step}: score {score:.1f}")
        step += 1
    return agent, score, step, val_losses, pol_losses, action_locs

def train(n_episodes,
          env_hparams, agent_hparams, actor_hparams, critic_hparams,
          n_steps = 300, initial_state = None):
    env = InvasiveEnv(env_hparams, initial_state)
    agent = Exerminator(env.side_len, actor_hparams, critic_hparams, agent_hparams)
    episode_lens = []
    episode_rwds = []
    val_losses, pol_losses, action_locs = [], [], []
    for e in range(n_episodes):
        agent, ep_score, ep_len, ep_val_losses, ep_pol_losses, ep_action_locs = \
            run_episode(agent, env, n_steps, e, initial_state)
        episode_lens.append(ep_len)
        episode_rwds.append(ep_score)
        val_losses.extend(ep_val_losses)
        pol_losses.extend(ep_pol_losses)
        action_locs.extend(ep_action_locs)

    return episode_lens, episode_rwds, val_losses, pol_losses, action_locs
