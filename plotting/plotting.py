import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

from utils import grid_to_locs
from settings import run_hparams


def plot_episode_stats(
        episode_lengths,
        episode_scores,
        val_losses,
        pol_losses,
        actions,
        populations,
        smoothing_window=100,
        show=True,
        save=False,
        save_dir='plots'
):

    f1 = plot_episode_length_over_time(episode_lengths, save, save_dir, show)
    f2 = plot_episode_reward_over_time(episode_scores, save, save_dir, show, smoothing_window)
    f3 = plot_episode_vs_timestep(episode_lengths, save, save_dir, show)
    f4 = plot_value_loss_over_time(save, save_dir, show, val_losses)
    f5 = plot_policy_loss_over_time(pol_losses, save, save_dir, show)
    f6 = plot_number_of_actions_over_time(actions, save, save_dir, show)
    f7 = plot_population_size_over_time(populations, save, save_dir, show)
    f8 = plot_occupied_cells_over_time(populations, save, save_dir, show)
    f9, images = plot_heatmaps_over_time(populations, actions, run_hparams['snapshot_interval'], save, save_dir, show)

    return f1 ,f2, f3, f4, f5, f6, f7, f8, f9, images


def plot_occupied_cells_over_time(populations, save, save_dir, show):
    # get number of occupied cells over time (count the number of non-zero cells)
    occupied_cells = [np.count_nonzero(pop) for pop in populations]
    # Plot number of occupied cells
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(occupied_cells)), occupied_cells)
    plt.xlabel("Steps")
    plt.ylabel("# of occupied cells")
    plt.title("Number of occupied cells over time steps")
    if save:
        plt.savefig(save_dir + '/occupied_cells.png')
    if not show:
        plt.close(fig)
    else:
        plt.show(fig)
    return fig


def plot_population_size_over_time(populations, save, save_dir, show):
    # get population size over time
    pop_sizes = [pop.sum().item() for pop in populations]

    # Plot population size
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(pop_sizes)), pop_sizes)
    plt.xlabel("Steps")
    plt.ylabel("Population size")
    plt.title("Population size over time steps")
    if save:
        plt.savefig(save_dir + '/pop_size.png')
    if not show:
        plt.close(fig)
    else:
        plt.show(fig)
    return fig


def plot_number_of_actions_over_time(action_locs, save, save_dir, show):
    # get number of action locations over time
    actions = [a.sum().item() for a in action_locs]

    # Plot number of action locations
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(actions)), actions)
    plt.xlabel("Steps")
    plt.ylabel("# action locations")
    plt.title("Number of action locations over time steps")
    if save:
        plt.savefig(save_dir + '/action_locations.png')
    if not show:
        plt.close(fig)
    else:
        plt.show(fig)
    return fig


def plot_policy_loss_over_time(pol_losses, save, save_dir, show):
    # Plot policy loss
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(pol_losses)), pol_losses)
    plt.xlabel("Steps")
    plt.ylabel("Policy Loss")
    plt.title("Policy loss over time steps")
    if save:
        plt.savefig(save_dir + '/policy_loss.png')
    if not show:
        plt.close(fig)
    else:
        plt.show(fig)
    return fig


def plot_value_loss_over_time(save, save_dir, show, val_losses):
    # Plot value loss
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.xlabel("Steps")
    plt.ylabel("Value Loss")
    plt.title("Value loss over time steps")
    if save:
        plt.savefig(save_dir + '/value_loss.png')
    if not show:
        plt.close(fig)
    else:
        plt.show(fig)
    return fig


def plot_episode_vs_timestep(episode_lengths, save, save_dir, show):
    # Plot time steps and episode number
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(episode_lengths), np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if save:
        plt.savefig(save_dir + '/episode_per_time_step.png')
    if not show:
        plt.close(fig)
    else:
        plt.show(fig)
    return fig


def plot_episode_reward_over_time(episode_scores, save, save_dir, show, smoothing_window):
    # Plot the episode reward over time
    fig = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(episode_scores).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if save:
        plt.savefig(save_dir + '/rewards.png')
    if not show:
        plt.close(fig)
    else:
        plt.show(fig)
    return fig


def plot_episode_length_over_time(episode_lengths, save, save_dir, show):
    # Plot the episode length over time
    fig = plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if save:
        plt.savefig(save_dir + '/episode_lengths.png')
    if not show:
        plt.close(fig)
    else:
        plt.show(fig)
    return fig


def plot_heatmaps_over_time(populations, actions, snapshot_interval, save, save_dir, show):
    images = []
    f, ax = plt.subplots()
    if save:
        os.makedirs(save_dir + '/heatmaps', exist_ok=True)

    for i in range(len(populations)):
        if i % snapshot_interval == 0:
            # Plot heatmaps of the population over time
            action_coords = grid_to_locs(actions[i])
            hmap = ax.imshow(populations[i], cmap='hot', interpolation='nearest', animated=True)
            dots = ax.scatter(action_coords[:, 0], action_coords[:, 1], c='lightgreen', s=10)
            title = ax.text(0.5, 0.85, f"Step {i}", c='r')
            images.append([hmap, dots, title])

    anim = make_animation(f, images, interval=200, repeat_delay=1000)
    if save:
        anim.save(save_dir + '/heatmaps/heatmap_animation.gif', writer='pillow', fps=5)
    if not show:
        plt.close(f)
    else:
        plt.show(f)
    return f, images

def make_animation(figure, images, interval, repeat_delay):
    # Make animation
    return animation.ArtistAnimation(figure, images, interval=interval, repeat_delay=repeat_delay, blit=False)
