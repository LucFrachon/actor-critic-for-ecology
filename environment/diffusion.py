import numpy as np

from environment.utils import locs_to_grid, grid_to_locs


def diffuse(grid, sigma_diff=1., clamp=True):
    locs = grid_to_locs(grid)
    # Update the particles' locations at random.
    mvmt = np.round(sigma_diff * np.random.standard_normal(locs.shape)).astype('int')
    locs += mvmt
    # Clamp the locations to the grid to prevent the particles from escaping.
    if clamp:
        locs = np.clip(locs, 0, grid.shape[0] - 1)
    # Create an updated grid and plot it.
    grid = locs_to_grid(locs, grid.shape[0])
    return grid


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import animation
    # Unit test
    # grid side length
    grid_size = 11
    # Maximum numbter of iterations.
    n_iterations = 100
    # Number of particles in the simulation.
    nparticles = 1000


    locs = np.ones((nparticles, 2), dtype=int) * grid_size // 2
    grid = locs_to_grid(locs, grid_size)
    print(grid)
    grids = []
    for _ in range(n_iterations):
        grid = diffuse(grid, .5, clamp=True)
        grids.append(grid)

    # Make a GIF of the diffusion process from the grids.
    fig, ax = plt.subplots()
    # Use a cold to hot colour map
    im = ax.imshow(grids[0], cmap='coolwarm')

    def update(i):
        im.set_data(grids[i])
        return im,

    ani = animation.FuncAnimation(fig, update, frames=len(grids), interval=100, blit=True)
    # Save animation as gif
    ani.save('../plots/diffusion.gif', writer='imagemagick', fps=10)
