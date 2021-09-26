import matplotlib.pyplot as plt


def main():
    data = {}
    with open("D:/_swmm_results/2021-09-23_18-44-53/costs.csv") as f:
        f.readline()
        for line in f:
            generation, ranking, *costs = line.split(",")

            if ranking != "0":
                continue

            data[int(generation)] = list(map(float, costs))

    fig, axs = plt.subplots(2, 4)  # Create a figure containing a single axes.

    axs[0, 0].plot(data.keys(), [costs[0] for costs in data.values()])
    axs[0, 0].set_title("total")

    axs[0, 1].plot(data.keys(), [costs[1] for costs in data.values()], label="cluster_count")
    axs[0, 1].set_title("cluster_count")

    axs[0, 2].plot(data.keys(), [costs[2] for costs in data.values()], label="magnet")
    axs[0, 2].set_title("magnet")

    axs[1, 0].plot(data.keys(), [costs[3] for costs in data.values()], label="area")
    axs[1, 0].set_title("area")

    axs[1, 1].plot(data.keys(), [costs[4] for costs in data.values()], label="repulsion")
    axs[1, 1].set_title("repulsion")

    axs[1, 2].plot(data.keys(), [costs[5] for costs in data.values()], label="attraction")
    axs[1, 2].set_title("attraction")

    axs[1, 3].plot(data.keys(), [costs[6] for costs in data.values()], label="runoff")
    axs[1, 3].set_title("runoff")


    # fig, axs = plt.subplots(1, 2)

    # axs[0].plot(data.keys(), [costs[0] for costs in data.values()])
    # axs[0].set_title("total")

    # axs[1].plot(data.keys(), [costs[6] for costs in data.values()], label="runoff")
    # axs[1].set_title("runoff")

    plt.show()


if __name__ == "__main__":
    main()
