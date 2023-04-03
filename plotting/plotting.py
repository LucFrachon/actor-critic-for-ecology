import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_episode_stats(episode_lengths, episode_scores, val_losses, pol_losses, action_locs, pop_sizes, occupied_cells,
                       smoothing_window=100, show=True, save=False, save_dir='plots'):
    # Plot the episode length over time
    ep_length = plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if save:
        plt.savefig(save_dir + '/episode_lengths.png')
    if not show:
        plt.close(ep_length)
    else:
        plt.show(ep_length)

    # Plot the episode reward over time
    ep_reward = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(episode_scores).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if save:
        plt.savefig(save_dir + '/rewards.png')
    if not show:
        plt.close(ep_reward)
    else:
        plt.show(ep_reward)

    # Plot time steps and episode number
    ep_time_steps = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(episode_lengths), np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if save:
        plt.savefig(save_dir + '/episode_per_time_step.png')
    if not show:
        plt.close(ep_time_steps)
    else:
        plt.show(ep_time_steps)

    # Plot value loss
    value_loss = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.xlabel("Steps")
    plt.ylabel("Value Loss")
    plt.title("Value loss over time steps")
    if save:
        plt.savefig(save_dir + '/value_loss.png')
    if not show:
        plt.close(value_loss)
    else:
        plt.show(value_loss)

    # Plot policy loss
    policy_loss = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(pol_losses)), pol_losses)
    plt.xlabel("Steps")
    plt.ylabel("Policy Loss")
    plt.title("Policy loss over time steps")
    if save:
        plt.savefig(save_dir + '/policy_loss.png')
    if not show:
        plt.close(policy_loss)
    else:
        plt.show(policy_loss)

    # Plot number of action locations
    n_action_locs = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(action_locs)), action_locs)
    plt.xlabel("Steps")
    plt.ylabel("# action locations")
    plt.title("Number of action locations over time steps")
    if save:
        plt.savefig(save_dir + '/action_locations.png')
    if not show:
        plt.close(n_action_locs)
    else:
        plt.show(n_action_locs)

    # Plot population size
    pop_size = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(pop_sizes)), pop_sizes)
    plt.xlabel("Steps")
    plt.ylabel("Population size")
    plt.title("Population size over time steps")
    if save:
        plt.savefig(save_dir + '/pop_size.png')
    if not show:
        plt.close(pop_size)
    else:
        plt.show(pop_size)

    # Plot number of occupied cells
    n_occupied_cells = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(occupied_cells)), occupied_cells)
    plt.xlabel("Steps")
    plt.ylabel("# of occupied cells")
    plt.title("Number of occupied cells over time steps")
    if save:
        plt.savefig(save_dir + '/occupied_cells.png')
    if not show:
        plt.close(n_occupied_cells)
    else:
        plt.show(n_occupied_cells)

    return ep_length, ep_reward, ep_time_steps, value_loss, policy_loss, pop_size, n_occupied_cells
