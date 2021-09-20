from ga2d import GeneticAlgorithm, GeneGenerator, Chromosome
from plotutils import plot_site


HEIGHT = 111
WIDTH = 71

TAG_TO_CODE = {
    "공동주택": 1,
    "주상복합": 2,
    "근린생활시설": 3,
    "상업시설": 4,
    "유보형복합용지": 5,
    "자족복합용지": 6,
    "자족시설": 7,
    "업무시설": 8,
    "공원": 9,
    "녹지": 10,
    "공공공지": 11,
    "보행자전용도로": 12,
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
        "Up": create_2dlist(),
        "Down": create_2dlist(),
        "Left": create_2dlist(),
        "Right": create_2dlist()
    }

    with open(path) as f:
        f.readline()

        for line in f:
            key, _, _, tag, _, _, _, area, *_ = line.strip().split(",")

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

                original_map[r][c] = TAG_TO_CODE[tag]
                mask[r][c] = 1
                area_map[r][c] = float(area)
                direction_masks["Up"][r][c] = up
                direction_masks["Down"][r][c] = down
                direction_masks["Left"][r][c] = left
                direction_masks["Right"][r][c] = right

    return original_map, mask, area_map, road_mask, road_area_map, neargreen_mask, direction_masks


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
COMMERCIAL_CORE1_OFFSET = (3, 4)
COMMERCIAL_CORE2 = (68, 48)
COMMERCIAL_CORE2_OFFSET = (7, 7)

def create_commercial_data():
    commercial_core_mask = create_2dlist()
    commercial_region_mask = create_2dlist()

    comm1_r, comm1_c = COMMERCIAL_CORE1
    comm2_r, comm2_c = COMMERCIAL_CORE2

    for dr, dc in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
        commercial_core_mask[comm1_r + dr][comm1_c + dc] = 1
        commercial_core_mask[comm2_r + dr][comm2_c + dc] = 1

    comm1_offset_r, comm1_offset_c = COMMERCIAL_CORE1_OFFSET
    comm2_offset_r, comm2_offset_c = COMMERCIAL_CORE2_OFFSET

    for r in range(comm1_r - comm1_offset_r, comm1_r + comm1_offset_r + 1):
        for c in range(comm1_c - comm1_offset_c, comm1_c + comm1_offset_c + 1):
            commercial_region_mask[r][c] = 1

    for r in range(comm2_r - comm2_offset_r, comm2_r + comm2_offset_r + 1):
        for c in range(comm2_c - comm2_offset_c, comm2_c + comm2_offset_c + 1):
            commercial_region_mask[r][c] = 1

    return commercial_core_mask, commercial_region_mask


BUSINESS_CORE1 = (80, 42)
BUSINESS_CORE1_OFFSET = (5, 5)
BUSINESS_CORE2 = (72, 32)
BUSINESS_CORE2_OFFSET = (2, 2)

def create_business_data():
    business_core_mask = create_2dlist()
    business_region_mask = create_2dlist()

    busi1_r, busi1_c = BUSINESS_CORE1
    busi2_r, busi2_c = BUSINESS_CORE2

    for dr, dc in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
        business_core_mask[busi1_r + dr][busi1_c + dc] = 1
        business_core_mask[busi2_r + dr][busi2_c + dc] = 1

    busi1_offset_r, busi1_offset_c = BUSINESS_CORE1_OFFSET
    busi2_offset_r, busi2_offset_c = BUSINESS_CORE2_OFFSET

    for r in range(busi1_r - busi1_offset_r, busi1_r + busi1_offset_r + 1):
        for c in range(busi1_c - busi1_offset_c, busi1_c + busi1_offset_c + 1):
            business_region_mask[r][c] = 1

    for r in range(busi2_r - busi2_offset_r, busi2_r + busi2_offset_r + 1):
        for c in range(busi2_c - busi2_offset_c, busi2_c + busi2_offset_c + 1):
            business_region_mask[r][c] = 1

    return business_core_mask, business_region_mask


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
    original_map, mask, area_map, road_mask, road_area_map, neargreen_mask, direction_masks = load_site_data("sub_og.csv")
    big_road_mask = create_big_road_mask(road_area_map)

    commercial_core_mask, commercial_region_mask = create_commercial_data()
    business_core_mask, business_region_mask = create_business_data()

    quiet_region_mask = create_quiet_region_mask()


    generator = GeneGenerator(HEIGHT, WIDTH, list(TAG_TO_CODE.values()))

    generator.add_mask(mask)
    generator.add_cluster_rule(8, 8)
    generator.add_area_rule([MAX_AREAS[CODE_TO_TAG[code]] for code in TAG_TO_CODE.values()])
    generator.change_area_map(area_map)

    generator.add_submask(1, quiet_region_mask)
    generator.add_submask(4, commercial_region_mask)
    generator.add_submask(8, business_region_mask)

    generator.add_magnet(4, commercial_core_mask, 20)
    generator.add_magnet(8, business_core_mask, 20)

    generator.add_magnet(9, neargreen_mask, 10)
    generator.add_magnet(10, neargreen_mask, 10)
    generator.add_magnet(11, neargreen_mask, 10)
    generator.add_magnet(12, neargreen_mask, 10)

    generator.add_magnet(2, big_road_mask, 20)
    generator.add_magnet(5, big_road_mask, 20)
    generator.add_magnet(6, big_road_mask, 20)
    generator.add_magnet(7, big_road_mask, 20)

    generator.add_repulsion_rule(1, 2)
    generator.add_repulsion_rule(1, 4)
    generator.add_repulsion_rule(1, 5)
    generator.add_repulsion_rule(1, 6)
    generator.add_repulsion_rule(1, 7)
    generator.add_repulsion_rule(1, 8)


    # genes = generator.generate()
    # chromosome = Chromosome(genes, generator)

    # print(chromosome.genes.analyze_cluster())
    # generator.evaluate(chromosome.genes)
    # print(chromosome.cost, chromosome._costs)

    # plot_site(chromosome.genes.raw)


    ga = GeneticAlgorithm(generator)
    best = ga.run(size=20, elitism=2)
    print(best.genes.analyze_cluster()[1])
    print(best._costs)
    plot_site(best.genes.raw)


if __name__ == "__main__":
    main()
