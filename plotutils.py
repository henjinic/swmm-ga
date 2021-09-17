import matplotlib.pyplot as plt


def plot(*args):
    if len(args) == 1:
        fig, ax = plt.subplots()
        img = ax.imshow(args[0], cmap="gist_ncar")
        fig.colorbar(img, ax=ax, aspect=50)
    else:
        fig, axs = plt.subplots(1, len(args))

        for ax, grid in zip(axs, args):
            img = ax.imshow(grid, cmap="gist_ncar")
            fig.colorbar(img, ax=ax, aspect=50)

    plt.show()


def main():
    pass

if __name__ == "__main__":
    main()