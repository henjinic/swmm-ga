# -*- coding: utf-8 -*-
from collections import defaultdict


HEIGHT = 111
WIDTH = 71
TAG_TO_CODE = {
    "공동주택": 1,
    "주상복합": 2,
    "근린생활시설": 3,
    "상업시설1": 4,
    "유보형복합용지": 5,
    "자족복합용지": 6,
    "자족시설": 7,
    "업무시설1": 8,
    "공원": 9,
    "녹지": 10,
    "공공공지": 11,
    "보행자전용도로": 12,
    "상업시설2": 13,
    "업무시설2": 14,
    "근생용지": 15,
    }
CODE_TO_TAG = {TAG_TO_CODE[tag]: tag for tag in TAG_TO_CODE}
NEARGREEN_TAGS = ["커뮤니티시설", "교육시설"]


def create_2dlist(height=HEIGHT, width=WIDTH, initial=0):
    return [[initial] * width for _ in range(height)]


def load_site_data(path):
    original_map = create_2dlist()
    mask = create_2dlist()
    area_map = create_2dlist()
    road_mask = create_2dlist()
    road_area_map = create_2dlist()
    neargreen_mask = create_2dlist()
    direction_vectors = defaultdict(list)
    # direction_masks = {
    #     "up": create_2dlist(),
    #     "down": create_2dlist(),
    #     "left": create_2dlist(),
    #     "right": create_2dlist()
    #     }
    original_areas = defaultdict(float)

# 3
    with open(path, encoding="utf-8-sig") as f:
        f.readline()

        for line in f:
            key, _, _, tag, _, _, area, *_ = line.strip().split(",")
# 2
    # with open(path) as f:
    #     f.readline()

    #     for line in f:
    #         words = line.strip().split(",")
    #         key = words[0]
    #         tag = words[3].decode("utf-8")
    #         area = words[6]
#
            if key[1] == "D":
                r = int(key[2:5])
                c = int(key[5:7])
                road_mask[r][c] = 1
                road_area_map[r][c] = float(area)

                continue

            if key[1] == "N":
                r = int(key[2:5])
                c = int(key[5:7])

                if tag in NEARGREEN_TAGS:
                    neargreen_mask[r][c] = 1
            else:
                r = int(key[1:4])
                c = int(key[4:6])
                # up, right, down, left = map(lambda x: 1 if x == "T" else 0, key[6:10])
                # up, right, down, left = map(lambda x: 1 if x == "T" else 0, key[6:10])
                up, right, down, left = (truth == "T" for truth in key[6:10])

                if tag == "상업시설":
                    tag += "1" if r < 40 else "2"
                elif tag =="업무시설":
                    tag += "1" if c < 40 else "2"

                original_map[r][c] = TAG_TO_CODE[tag]
                mask[r][c] = 1
                area_map[r][c] = float(area)
                if up:
                    direction_vectors[r, c].append((-1, 0))
                if right:
                    direction_vectors[r, c].append((0, 1))
                if down:
                    direction_vectors[r, c].append((1, 0))
                if left:
                    direction_vectors[r, c].append((0, -1))
                # direction_masks["up"][r][c] = up
                # direction_masks["down"][r][c] = down
                # direction_masks["left"][r][c] = left
                # direction_masks["right"][r][c] = right
                original_areas[original_map[r][c]] += area_map[r][c]

    # original_areas = [v for k, v in sorted(original_areas.items())]

    return original_map, mask, area_map, road_mask, road_area_map, neargreen_mask, direction_vectors, original_areas


def create_big_road_mask(road_area_map):
    big_road_mask = create_2dlist()

    for r in range(HEIGHT):
        for c in range(WIDTH):
            vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            is_valid_coord = lambda r, c: 0 <= r < HEIGHT and 0 <= c < WIDTH
            total_area = sum([road_area_map[r + dr][c + dc] for dr, dc in vectors if is_valid_coord(r + dr, c + dc)])

            if total_area >= 0.14:
                big_road_mask[r][c] = 1

    return big_road_mask


def create_core_region_data(core, offset):
    core_mask = create_2dlist()
    region_mask = create_2dlist()

    core_mask[core[0]][core[1]] = 1

    for r in range(core[0] - offset[0], core[0] + offset[0] + 1):
        for c in range(core[1] - offset[1], core[1] + offset[1] + 1):
            region_mask[r][c] = 1

    return core_mask, region_mask


def create_quiet_region_mask(point1, point2):
    quiet_region_mask = create_2dlist()

    for r in range(HEIGHT):
        for c in range(WIDTH):
            r1, c1 = point1
            r2, c2 = point2
            if (r2 - r1) * c + r1 * c2 <= (c2 - c1) * r + r2 * c1:
                quiet_region_mask[r][c] = 1

    return quiet_region_mask


def main():
    pass


if __name__ == "__main__":
    main()
