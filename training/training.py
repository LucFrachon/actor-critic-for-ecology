import time
import numpy as np
import torch
from environment.environment import InvasiveEnv
from agent.agent import Exerminator


def run_episode(agent, env, n_steps, ep_idx, initial_state = None, print_every = 25):
    """
    :param initial_state: A (n_population, 2) np.ndarray with the initial positions of each population member.
        If not provided, starts with n_pop_ini in the centre of the grid.
    """
    step, score, is_terminal = 0, 0, False
    val_loss_over_ep, pol_loss_over_ep = [], []
    actions_over_ep = []
    population_over_ep = []
    agent.current_state = torch.tensor(env.reset(initial_state), device=agent.device).reshape((
        1, 1, env.side_len, env.side_len))
    if agent.hparams.normalise_states:
        agent.current_state = agent.normalise_state(agent.current_state)  # normalise
    while (step < n_steps) and (not is_terminal):
        action = agent.pick_action()
        actions_over_ep.append(action)
        next_state, reward, is_terminal = env.step(action)
        agent.observe(next_state, reward, is_terminal)
        val_loss, pol_loss = agent.train()
        val_loss_over_ep.append(val_loss)
        pol_loss_over_ep.append(pol_loss)
        population = env.grid
        population_over_ep.append(population)
        score += reward
        if step % print_every == 0:
            n_actions = int(action.sum().item())
            pop_size = int(population.sum().item())
            occupancy = int(np.count_nonzero(population))
            print(
                f"\rEpisode {ep_idx}, Step {step}: Acted on {n_actions} cells; "
                f"last step reward {reward:.2f}, current score {score:.1f}, "
                f"population {pop_size}, occupied cells: {occupancy}")
        if is_terminal:
            print("New episode starting...")
            break
        step += 1
    return agent, score, step, val_loss_over_ep, pol_loss_over_ep, actions_over_ep, population_over_ep

def train(n_episodes,
          env_hparams, agent_hparams, actor_hparams, critic_hparams,
          n_steps = 300, initial_state = None, print_every = 25):
    start = time.time()
    env = InvasiveEnv(env_hparams, initial_state)
    agent = Exerminator(env.side_len, actor_hparams, critic_hparams, agent_hparams)
    episode_lens, episode_rwds = [], []
    val_losses, pol_losses, actions = [], [], []
    populations = []

    for e in range(n_episodes):
        agent, ep_score, ep_len, ep_val_losses, ep_pol_losses, ep_action_locs, pop_over_ep = \
            run_episode(agent, env, n_steps, e, initial_state, print_every=print_every)
        episode_lens.append(ep_len)
        episode_rwds.append(ep_score)
        val_losses.extend(ep_val_losses)
        pol_losses.extend(ep_pol_losses)
        actions.extend(ep_action_locs)
        populations.extend(pop_over_ep)
    print(f'Training completed in {time.time() - start} seconds.')

    return episode_lens, episode_rwds, val_losses, pol_losses, actions, populations
