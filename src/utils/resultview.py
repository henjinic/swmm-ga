import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.colors import ListedColormap

from dataloader import load_site_data, CODE_TO_TAG
from gridutils import analyze_cluster


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


def view_full(original_map, new_map, neargreen_map):
    show_two_map(original_map + neargreen_map, new_map + neargreen_map)


def view_area_change(original_map, new_map, area_map):
    original_areas = defaultdict(float)
    new_areas = defaultdict(float)
    for code in range(1, 16):
        original_areas[CODE_TO_TAG[code].strip("12")] += area_map[original_map == code].sum()
        new_areas[CODE_TO_TAG[code].strip("12")] += area_map[new_map == code].sum()
    for tag in original_areas:
        rate = (new_areas[tag] - original_areas[tag]) * 100 / original_areas[tag]
        print(tag, f"{original_areas[tag]:.2f}", f"{new_areas[tag]:.2f}", f"{rate:.2f}%")


def view_cluster_count(original_map, new_map):
    original_count = sum(len(clusters) for clusters in analyze_cluster(original_map).values())
    new_count = sum(len(clusters) for clusters in analyze_cluster(new_map).values())
    print(original_count, new_count)


def view_road_core(original_map, new_map, road_map):
    def road_core_map(source_map, road_map):
        result = road_map.copy()
        result[result == 1] = -1
        result[(source_map == 4) | (source_map == 13)] = 4
        result[(source_map == 8) | (source_map == 14)] = 8
        return result

    map1 = road_core_map(original_map, road_map)
    map2 = road_core_map(new_map, road_map)
    show_two_map(map1, map2)


def view_house(original_map, new_map):
    original_result = original_map.copy()
    original_result[original_result != 1] = 0

    new_result = new_map.copy()
    new_result[new_result != 1] = 0

    show_two_map(original_result, new_result)


def view_road_commercial(original_map, new_map, road_map):
    def road_commercial_map(source_map, road_map):
        result = road_map.copy()
        result[result == 1] = -1
        result[(source_map == 4) | (source_map == 13)] = 4
        result[source_map == 5] = 5
        result[source_map == 6] = 6
        result[source_map == 7] = 7
        result[(source_map == 8) | (source_map == 14)] = 8
        return result

    map1 = road_commercial_map(original_map, road_map)
    map2 = road_commercial_map(new_map, road_map)
    show_two_map(map1, map2)


def view_green_cultural(original_map, new_map, neargreen_map):
    def green_cultural_map(source_map, neargreen_map):
        result = source_map.copy()
        mask = (9 <= source_map) & (source_map <= 12)
        result[~mask] = 0
        return result + neargreen_map

    map1 = green_cultural_map(original_map, neargreen_map)
    map2 = green_cultural_map(new_map, neargreen_map)
    show_two_map(map1, map2)


def view_house_commercial(original_map, new_map):
    def house_commercial_map(source_map):
        result = source_map.copy()
        mask1 = (source_map == 1)
        mask2 = (source_map == 4) | (source_map == 13)
        mask3 = (source_map == 8) | (source_map == 14)
        mask4 = (5 <= source_map) & (source_map <= 7)
        result[~(mask1 | mask2 | mask3 | mask4)] = 0
        return result

    map1 = house_commercial_map(original_map)
    map2 = house_commercial_map(new_map)
    show_two_map(map1, map2)


def show_two_map(map1, map2):
    plt.subplot(121)
    plt.imshow(map1, cmap=SITE_CMAP, vmin=-1, vmax=17)
    plt.subplot(122)
    plt.imshow(map2, cmap=SITE_CMAP, vmin=-1, vmax=17)
    plt.show()


def main():
    site_data = load_site_data("../sub_og.csv")
    original_map = np.array(site_data[0])
    area_map = np.array(site_data[2])
    road_map = np.array(site_data[3])
    neargreen_map = np.array(site_data[8])
    new_map = np.loadtxt("D:/_swmm_results/2021-11-21_17-56-32/527/0.csv", delimiter=",")

    while True:
        print("0. exit")
        print("1. full")
        print("2. area change")
        print("3. cluster count")
        print("4. road - core")
        print("5. house")
        print("6. road - commercial")
        print("7. green - cultural")
        print("8. house - commercial")
        match int(input(">>> ")):
            case 0:
                break
            case 1:
                view_full(original_map, new_map, neargreen_map)
            case 2:
                view_area_change(original_map, new_map, area_map)
            case 3:
                view_cluster_count(original_map, new_map)
            case 4:
                view_road_core(original_map, new_map, road_map)
            case 5:
                view_house(original_map, new_map)
            case 6:
                view_road_commercial(original_map, new_map, road_map)
            case 7:
                view_green_cultural(original_map, new_map, neargreen_map)
            case 8:
                view_house_commercial(original_map, new_map)
        print()


if __name__ == "__main__":
    main()
