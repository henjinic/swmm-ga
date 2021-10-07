# -*- coding: utf-8 -*-
from dataloader import (create_big_road_mask, create_core_region_data, create_quiet_region_mask,
                        load_site_data, HEIGHT, TAG_TO_CODE, WIDTH)
from ga2d import Chromosome, GeneticAlgorithm, GeneRuler
from rules import (AreaMaxRule, AreaMinRule, AttractionRule,
                   ClusterCountMaxRule, MagnetRule, RepulsionRule)

# 3
from matplotlib import pyplot as plt
from plotutils import plot_site
#


COMMERCIAL_CORE1 = (13, 52)
COMMERCIAL_CORE_OFFSET1 = (3, 4)
COMMERCIAL_CORE2 = (68, 48)
COMMERCIAL_CORE_OFFSET2 = (7, 7)

BUSINESS_CORE1 = (72, 32)
BUSINESS_CORE_OFFSET1 = (2, 2)
BUSINESS_CORE2 = (80, 42)
BUSINESS_CORE_OFFSET2 = (5, 5)

QUIET_DIVIDE_POINT1 = (43, 53)
QUIET_DIVIDE_POINT2 = (48, 67)


def main():
    _, mask, area_map, _, road_area_map, neargreen_mask, direction_masks, original_areas = load_site_data("sub_og.csv")
    big_road_mask = create_big_road_mask(road_area_map)
    quiet_region_mask = create_quiet_region_mask(QUIET_DIVIDE_POINT1, QUIET_DIVIDE_POINT2)
    commercial_core_mask1, commercial_region_mask1 = create_core_region_data(COMMERCIAL_CORE1, COMMERCIAL_CORE_OFFSET1)
    commercial_core_mask2, commercial_region_mask2 = create_core_region_data(COMMERCIAL_CORE2, COMMERCIAL_CORE_OFFSET2)
    business_core_mask1, business_region_mask1 = create_core_region_data(BUSINESS_CORE1, BUSINESS_CORE_OFFSET1)
    business_core_mask2, business_region_mask2 = create_core_region_data(BUSINESS_CORE2, BUSINESS_CORE_OFFSET2)

    ruler = GeneRuler(HEIGHT, WIDTH, list(range(1, 16)), mask, direction_masks)

    ruler.add_rule(ClusterCountMaxRule(250))

    # # condition 1
    # ruler.add_submask(TAG_TO_CODE["상업시설1"], commercial_region_mask1)
    # ruler.add_rule(MagnetRule(TAG_TO_CODE["상업시설1"], commercial_core_mask1, 0, 1, "commercial_core1"))
    # ruler.add_magnet(TAG_TO_CODE["상업시설1"], commercial_core_mask1, 16)

    # ruler.add_submask(TAG_TO_CODE["상업시설2"], commercial_region_mask2)
    # ruler.add_magnet(TAG_TO_CODE["상업시설2"], commercial_core_mask2, 16)
    # ruler.add_rule(MagnetRule(TAG_TO_CODE["상업시설2"], commercial_core_mask2, 0, 1, "commercial_core2"))

    # ruler.add_submask(TAG_TO_CODE["업무시설1"], business_region_mask1)
    # ruler.add_magnet(TAG_TO_CODE["업무시설1"], business_core_mask1, 16)

    # ruler.add_submask(TAG_TO_CODE["업무시설2"], business_region_mask2)
    # ruler.add_magnet(TAG_TO_CODE["업무시설2"], business_core_mask2, 16)

    # # condition 2
    # ruler.add_submask(TAG_TO_CODE["공동주택"], quiet_region_mask)

    # # condition 3
    # ruler.add_magnet(TAG_TO_CODE["상업시설1"], big_road_mask, 4)
    # ruler.add_magnet(TAG_TO_CODE["상업시설2"], big_road_mask, 4)
    # ruler.add_magnet(TAG_TO_CODE["업무시설1"], big_road_mask, 4)
    # ruler.add_magnet(TAG_TO_CODE["업무시설2"], big_road_mask, 4)
    # ruler.add_magnet(TAG_TO_CODE["유보형복합용지"], big_road_mask, 4)
    # ruler.add_magnet(TAG_TO_CODE["자족복합용지"], big_road_mask, 4)
    # ruler.add_magnet(TAG_TO_CODE["자족시설"], big_road_mask, 4)

    # # condition 4 & 5
    # ruler.add_magnet(TAG_TO_CODE["녹지"], neargreen_mask, 4)
    # ruler.add_attraction_rule(TAG_TO_CODE["공원"], TAG_TO_CODE["녹지"])
    # ruler.add_attraction_rule(TAG_TO_CODE["공공공지"], TAG_TO_CODE["녹지"])
    # ruler.add_attraction_rule(TAG_TO_CODE["보행자전용도로"], TAG_TO_CODE["녹지"])

    # # condition 6
    # ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["상업시설1"])
    # ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["상업시설2"])
    # ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["업무시설1"])
    # ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["업무시설2"])
    # ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["유보형복합용지"])
    # ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["자족복합용지"])
    # ruler.add_repulsion_rule(TAG_TO_CODE["공동주택"], TAG_TO_CODE["자족시설"])


    parent1 = Chromosome(ruler.generate(), ruler)
    parent2 = Chromosome(ruler.generate(), ruler)
    print(parent1.cost_detail)
    print(parent2.cost_detail)

    child1, child2 = parent1.crossover(parent2)
    print(child1.cost_detail)
    print(child2.cost_detail)

    plt.set_cmap("rainbow")
    plt.subplot(141)
    plt.imshow(parent1.genes.raw)
    plt.subplot(142)
    plt.imshow(parent2.genes.raw)
    plt.subplot(143)
    plt.imshow(child1.genes.raw)
    plt.subplot(144)
    plt.imshow(child2.genes.raw)
    plt.show()



    # sites = []
    # for _ in range(2):
    #     chromosome = Chromosome(ruler.generate(), ruler)

    #     print("the number of clusters:", chromosome.genes.analyze_cluster()[1])
    #     # ruler.evaluate(chromosome.genes)
    #     print("total cost:", chromosome.cost)
    #     print(chromosome.cost_detail)
    #     sites.append(chromosome.genes.raw)

    # plot_site(*sites)


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
    # best = ga.run(size=2, strategy="cost", child_count=2, mutation_rate=0.9, stable_step_for_exit=2, is_logging=False)
    # # best = ga.age_based_run(size=4, elitism=2, mutation_rate=0.9, stable_step_for_exit=1)
    # print("the number of clusters:", best.genes.analyze_cluster()[1])
    # print("best chromosome details", best.cost_detail)
    # plot_site(best.genes.raw)


if __name__ == "__main__":
    main()
