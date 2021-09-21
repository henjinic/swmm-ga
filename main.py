from collections import defaultdict
from ga2d import GeneticAlgorithm, GeneRuler, Chromosome
from plotutils import plot_site


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
    "근생용지": 15
}

CODE_TO_TAG = {TAG_TO_CODE[tag]: tag for tag in TAG_TO_CODE}

MAX_AREAS = {
    "공동주택": 63.924,
    "주상복합": 14.3891,
    "근린생활시설": 3.99112,
    "상업시설": 2.98405,
    "유보형복합용지": 8.28045,
    "자족복합용지": 13.2233,
    "자족시설": 62.2402,
    "업무시설": 1.75321,
    "공원": 51.1226,
    "녹지": 12.1783,
    "공공공지": 2.56442,
    "보행자전용도로": 2.01097,
}

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
    direction_masks = {
        "up": create_2dlist(),
        "down": create_2dlist(),
        "left": create_2dlist(),
        "right": create_2dlist()
    }
    original_areas = defaultdict(float)

    with open(path, encoding="utf-8-sig") as f:
        f.readline()

        for line in f:
            key, _, _, tag, _, _, area, *_ = line.strip().split(",")

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
                up, right, down, left = map(lambda x: 1 if x == "T" else 0, key[6:10])

                if tag == "상업시설":
                    tag += "1" if r < 40 else "2"
                elif tag =="업무시설":
                    tag += "1" if c < 40 else "2"

                original_map[r][c] = TAG_TO_CODE[tag]
                mask[r][c] = 1
                area_map[r][c] = float(area)
                direction_masks["up"][r][c] = up
                direction_masks["down"][r][c] = down
                direction_masks["left"][r][c] = left
                direction_masks["right"][r][c] = right
                original_areas[original_map[r][c]] += area_map[r][c]

    original_areas = [v for k, v in sorted(original_areas.items())]

    return original_map, mask, area_map, road_mask, road_area_map, neargreen_mask, direction_masks, original_areas

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

COMMERCIAL_CORE1 = (13, 52)
COMMERCIAL_CORE_OFFSET1 = (3, 4)
COMMERCIAL_CORE2 = (68, 48)
COMMERCIAL_CORE_OFFSET2 = (7, 7)

BUSINESS_CORE1 = (72, 32)
BUSINESS_CORE_OFFSET1 = (2, 2)
BUSINESS_CORE2 = (80, 42)
BUSINESS_CORE_OFFSET2 = (5, 5)

def create_core_region_data(core, offset):
    core_mask = create_2dlist()
    region_mask = create_2dlist()

    # for dr, dc in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
    for dr, dc in [(0, 0)]:
        core_mask[core[0] + dr][core[1] + dc] = 1

    for r in range(core[0] - offset[0], core[0] + offset[0] + 1):
        for c in range(core[1] - offset[1], core[1] + offset[1] + 1):
            region_mask[r][c] = 1

    return core_mask, region_mask

QUIET_DIVIDE_POINT1 = (43, 53)
QUIET_DIVIDE_POINT2 = (48, 67)

def create_quiet_region_mask():
    quiet_region_mask = create_2dlist()

    for r in range(HEIGHT):
        for c in range(WIDTH):
            r1, c1 = QUIET_DIVIDE_POINT1
            r2, c2 = QUIET_DIVIDE_POINT2
            if (r2 - r1) * c + r1 * c2 <= (c2 - c1) * r + r2 * c1:
                quiet_region_mask[r][c] = 1

    return quiet_region_mask


def main():
    original_map, mask, area_map, road_mask, road_area_map, neargreen_mask, direction_masks, original_areas = load_site_data("new_sub.csv")
    big_road_mask = create_big_road_mask(road_area_map)
    quiet_region_mask = create_quiet_region_mask()
    commercial_core_mask1, commercial_region_mask1 = create_core_region_data(COMMERCIAL_CORE1, COMMERCIAL_CORE_OFFSET1)
    commercial_core_mask2, commercial_region_mask2 = create_core_region_data(COMMERCIAL_CORE2, COMMERCIAL_CORE_OFFSET2)
    business_core_mask1, business_region_mask1 = create_core_region_data(BUSINESS_CORE1, BUSINESS_CORE_OFFSET1)
    business_core_mask2, business_region_mask2 = create_core_region_data(BUSINESS_CORE2, BUSINESS_CORE_OFFSET2)

    ruler = GeneRuler(HEIGHT, WIDTH, list(TAG_TO_CODE.values()))

    ruler.add_mask(mask)
    ruler.add_cluster_rule(24, 8)
    ruler.add_direction_masks(direction_masks)
    ruler.add_area_rule(original_areas, change_rate=0.5)
    ruler.change_area_map(area_map)

    # condition 1
    ruler.add_submask(TAG_TO_CODE["상업시설1"], commercial_region_mask1)
    ruler.add_magnet(TAG_TO_CODE["상업시설1"], commercial_core_mask1, 16)

    ruler.add_submask(TAG_TO_CODE["상업시설2"], commercial_region_mask2)
    ruler.add_magnet(TAG_TO_CODE["상업시설2"], commercial_core_mask2, 16)

    ruler.add_submask(TAG_TO_CODE["업무시설1"], business_region_mask1)
    ruler.add_magnet(TAG_TO_CODE["업무시설1"], business_core_mask1, 16)

    ruler.add_submask(TAG_TO_CODE["업무시설2"], business_region_mask2)
    ruler.add_magnet(TAG_TO_CODE["업무시설2"], business_core_mask2, 16)

    # condition 2
    ruler.add_submask(TAG_TO_CODE["공동주택"], quiet_region_mask)

    # condition 3
    ruler.add_magnet(TAG_TO_CODE["상업시설1"], big_road_mask, 4)
    ruler.add_magnet(TAG_TO_CODE["상업시설2"], big_road_mask, 4)
    ruler.add_magnet(TAG_TO_CODE["업무시설1"], big_road_mask, 4)
    ruler.add_magnet(TAG_TO_CODE["업무시설2"], big_road_mask, 4)
    ruler.add_magnet(TAG_TO_CODE["유보형복합용지"], big_road_mask, 4)
    ruler.add_magnet(TAG_TO_CODE["자족복합용지"], big_road_mask, 4)
    ruler.add_magnet(TAG_TO_CODE["자족시설"], big_road_mask, 4)

    # condition 4 & 5
    ruler.add_magnet(TAG_TO_CODE["녹지"], neargreen_mask, 4)
    ruler.add_attraction_rule(TAG_TO_CODE["공원"], TAG_TO_CODE["녹지"])
    ruler.add_attraction_rule(TAG_TO_CODE["공공공지"], TAG_TO_CODE["녹지"])
    ruler.add_attraction_rule(TAG_TO_CODE["보행자전용도로"], TAG_TO_CODE["녹지"])

    # condition 6
    ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["상업시설1"])
    ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["상업시설2"])
    ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["업무시설1"])
    ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["업무시설2"])
    ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["유보형복합용지"])
    ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["자족복합용지"])
    ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["자족시설"])


    sites = []
    for _ in range(2):
        chromosome = Chromosome(ruler.generate(), ruler)

        print("the number of clusters:", chromosome.genes.analyze_cluster()[1])
        ruler.evaluate(chromosome.genes)
        print("total cost:", chromosome.cost)
        print(chromosome.cost_detail)
        sites.append(chromosome.genes.raw)

    plot_site(*sites)


    # grid1 = ruler.generate()
    # grid2 = ruler.generate()

    # parent1 = Chromosome(grid1, ruler)
    # parent2 = Chromosome(grid2, ruler)
    # child1, child2 = parent1.crossover(parent2)
    # print(child1.genes.direction_masks)
    # print(*[x.genes.analyze_cluster()[1] for x in [parent1, parent2, child1, child2]])
    # print(*[x.cost for x in [parent1, parent2, child1, child2]])
    # print(*[x._costs for x in [parent1, parent2, child1, child2]])

    # plot_site(parent1.genes.raw, parent2.genes.raw, child1.genes.raw, child2.genes.raw)


    # ga = GeneticAlgorithm(ruler)
    # best = ga.run(size=20, elitism=2, mutation_rate=0.1, step=10)
    # print(best.genes.analyze_cluster()[1])
    # print(best._costs)
    # # plot_site(best.genes.raw)


if __name__ == "__main__":
    main()
