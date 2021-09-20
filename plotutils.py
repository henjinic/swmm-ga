import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


SITE_CMAP = ListedColormap([
    "black",
    "orange",
    "chocolate",
    "wheat",
    "red",
    "lightsalmon",
    "palevioletred",
    "mediumslateblue",
    "royalblue",
    "lime",
    "palegreen",
    "olive",
    "green",
])


def plot_grid(*args):
    if len(args) == 1:
        fig, ax = plt.subplots()
        img = ax.imshow(args[0], cmap="gist_ncar")
        fig.colorbar(img, ax=ax, aspect=50)
    else:
        fig, axs = plt.subplots(1, len(args))

        for ax, grid in zip(axs, args):
            img = ax.imshow(grid.raw, cmap="gist_ncar")
            fig.colorbar(img, ax=ax, aspect=50)

    plt.show()


def plot_site(*args):
    if len(args) == 1:
        fig, ax = plt.subplots()
        img = ax.imshow(args[0], cmap=SITE_CMAP, vmin=0, vmax=12)
        fig.colorbar(img, ax=ax, aspect=50)
    else:
        fig, axs = plt.subplots(1, len(args))

        for ax, site in zip(axs, args):
            img = ax.imshow(site, cmap=SITE_CMAP, vmin=0, vmax=12)
            fig.colorbar(img, ax=ax, aspect=50)

    plt.show()


def main():
    pass

if __name__ == "__main__":
    main()