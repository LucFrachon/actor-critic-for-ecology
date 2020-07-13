import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_episode_stats(episode_lengths, episode_scores, val_losses, pol_losses, action_locs, pop_sizes, occupied_cells,
                       smoothing_window = 100, show=True, save=False, save_dir='plots'):

    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if save:
        plt.savefig(save_dir + '/episode_lengths.png')
    if not show:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(episode_scores).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if save:
        plt.savefig(save_dir + '/rewards.png')
    if not show:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(episode_lengths), np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if save:
        plt.savefig(save_dir + '/episode_per_time_step.png')
    if not show:
        plt.close(fig3)
    else:
        plt.show(fig3)

    # Plot value loss
    fig4 = plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.xlabel("Steps")
    plt.ylabel("Value Loss")
    plt.title("Value loss over time steps")
    if save:
        plt.savefig(save_dir + '/value_loss.png')
    if not show:
        plt.close(fig4)
    else:
        plt.show(fig4)

    # Plot policy loss
    fig5 = plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(pol_losses)), pol_losses)
    plt.xlabel("Steps")
    plt.ylabel("Policy Loss")
    plt.title("Policy loss over time steps")
    if save:
        plt.savefig(save_dir + '/policy_loss.png')
    if not show:
        plt.close(fig5)
    else:
        plt.show(fig5)

    # Plot number of action locations
    fig5 = plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(action_locs)), action_locs)
    plt.xlabel("Steps")
    plt.ylabel("# action locations")
    plt.title("Number of action locations over time steps")
    if save:
        plt.savefig(save_dir + '/action_locations.png')
    if not show:
        plt.close(fig5)
    else:
        plt.show(fig5)

    # Plot population size
    fig6 = plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(pop_sizes)), pop_sizes)
    plt.xlabel("Steps")
    plt.ylabel("Population size")
    plt.title("Population size over time steps")
    if save:
        plt.savefig(save_dir + '/pop_size.png')
    if not show:
        plt.close(fig6)
    else:
        plt.show(fig6)

    # Plot number of occupied cells
    fig7 = plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(occupied_cells)), occupied_cells)
    plt.xlabel("Steps")
    plt.ylabel("# of occupied cells")
    plt.title("Number of occupied cells over time steps")
    if save:
        plt.savefig(save_dir + '/occupied_cells.png')
    if not show:
        plt.close(fig7)
    else:
        plt.show(fig7)


    return fig1, fig2, fig3, fig4, fig5, fig6, fig7
