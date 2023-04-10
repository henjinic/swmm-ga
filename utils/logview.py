import matplotlib.pyplot as plt
from collections import defaultdict


def main():
    data = defaultdict(list)
    with open("D:/_swmm_results/2021-11-21_17-56-32/costs.csv") as f:
        cost_names = list(f.readline().strip().split(","))[2:]
        print(cost_names)
        for line in f:
            _, ranking, *costs = line.split(",")
            if ranking != "0":
                continue
            for name, value in zip(cost_names, costs):
                data[name].append(float(value))

    print("-1. exit")
    for i, cost_name in enumerate(cost_names):
        print(f"{i}. {cost_name}")

    while True:
        match int(input(">>> ")):
            case -1:
                break
            case 0:
                start = int(input("start="))
                plt.plot(range(len(data["total"]))[start:], data["total"][start:])
                plt.title("total")
                plt.show()
            case x if x < len(cost_names):
                plt.plot(range(len(data[cost_names[x]])), data[cost_names[x]])
                plt.title(cost_names[x])
                plt.show()


if __name__ == "__main__":
    main()
