import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


SITE_CMAP = ListedColormap([
    "white",        # street, -1
    "black",        # empty
    "orange",       # house
    "chocolate",
    "wheat",
    "red",
    "lightsalmon",
    "palevioletred",
    "mediumslateblue",
    "blue",
    "lime",
    "palegreen",
    "olive",
    "green",
    "red",
    "blue",
    "magenta",
    "cyan",
    "skyblue"
])


def main():
    df = pd.read_csv("D:/_swmm_results/2021-11-21_17-56-32/costs.csv")

    header = [
        "generation",
        "rank",
        "total",
        "total_runoff",
        "cluster_count",
        "magnet",
        "area",
        "repulsion",
        "attraction",
    ]

    for prefix in header[4:]:
        df[f"{prefix}"] = df.filter(regex=f"^{prefix}").sum(axis=1)

    df = df[header]
    df = df.sort_values("generation", ascending=False, kind="stable")

    indeces = df.idxmin()

    for target in header[2:]:
        row = df.loc[indeces[target]]
        generation = int(row["generation"])
        rank = int(row["rank"])

        map = np.loadtxt(f"D:/_swmm_results/2021-11-21_17-56-32/{generation}/{rank}.csv", delimiter=",")
        plt.suptitle(f"{target.replace('_', ' ').capitalize()} (Generation: {generation}, Rank: {rank})")
        plt.subplot(121)
        plt.imshow(map, cmap=SITE_CMAP, vmin=-1, vmax=17)
        plt.subplot(122)
        plt.bar(header[2:], row[header[2:]])
        for i, col in enumerate(header[2:]):
            plt.text(i, row[col] + 2000, f"{row[col]:.2f}", ha="center")
        plt.show()


if __name__ == "__main__":
    main()
