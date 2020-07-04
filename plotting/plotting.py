import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_episode_stats(episode_lengths, episode_scores, val_losses, pol_losses, action_locs,
                       smoothing_window = 100, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
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
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(episode_lengths), np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    # Plot value loss
    fig4 = plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.xlabel("Steps")
    plt.ylabel("Value Loss")
    plt.title("Value loss over time steps")
    if noshow:
        plt.close(fig4)
    else:
        plt.show(fig4)

    # Plot policy loss
    fig5 = plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(pol_losses)), pol_losses)
    plt.xlabel("Steps")
    plt.ylabel("Policy Loss")
    plt.title("Policy loss over time steps")
    if noshow:
        plt.close(fig5)
    else:
        plt.show(fig5)

    # Plot number of action locations
    fig5 = plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(action_locs)), action_locs)
    plt.xlabel("Steps")
    plt.ylabel("# action locations")
    plt.title("Number of action locations over time steps")
    if noshow:
        plt.close(fig5)
    else:
        plt.show(fig5)

    return fig1, fig2, fig3, fig4, fig5
