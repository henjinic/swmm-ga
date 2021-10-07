# -*- coding: utf-8 -*-
import random
from operator import attrgetter

from logger import GALogger27
from mathutils import Grid, lerp
from randutils import choices, randpop
from rules import ClusterCountMaxRule


class Chromosome:

    def __init__(self, genes, gene_ruler):
        self._genes = genes
        self._gene_ruler = gene_ruler
        self._costs = None

    @property
    def genes(self):
        return self._genes

    @property
    def cost(self):
        return self.cost_detail["total"]

    @property
    def cost_detail(self):
        if self._costs is None:
            self._evaluate()
        return self._costs

    def crossover(self, partner):
        child_genes1 = self._genes.copy()
        child_genes2 = self._genes.copy()

        diff_coords = self._genes.get_diff_coords(partner._genes)

        for (gene1, gene2), coords in diff_coords.items():
            mask = Grid(height=self.genes.height, width=self.genes.width)
            for r, c in coords:
                mask[r, c] = 1
                child_genes1[r, c] = 0
                child_genes2[r, c] = 0

            self._gene_ruler.fill(child_genes1, mask, [gene1, gene2])
            self._gene_ruler.fill(child_genes2, mask, [gene1, gene2])

        return Chromosome(child_genes1, self._gene_ruler), Chromosome(child_genes2, self._gene_ruler)

    def mutate(self):
        region_height = self.genes.height // 5
        region_width = self.genes.width // 5
        r_start = random.randint(0, self.genes.height - region_height)
        c_start = random.randint(0, self.genes.width - region_width)

        region_mask = Grid(height=self.genes.height, width=self.genes.width)

        for r in range(r_start, r_start + region_height):
            for c in range(c_start, c_start + region_width):
                region_mask[r, c] = 1
                self.genes[r, c] = 0

        self._gene_ruler.fill(self.genes, region_mask)
        self._cost = None

    def _evaluate(self):
        self._costs = self._gene_ruler.evaluate(self._genes)


class GeneticAlgorithm:

    def __init__(self, gene_ruler):
        self._gene_ruler = gene_ruler

    def run(self, size=20, strategy="age", elitism=2, child_count=20,
            mutation_rate=0.05, stable_step_for_exit=20, is_logging=True):
        if is_logging:
            logger = GALogger27("D:/_swmm_results", "now")

        population = self._initialize(size)
        generation = 1

        print(generation, ">>>", [int(x.cost) for x in population])
        print(population[0].cost_detail)

        if is_logging:
            for i, chromosome in enumerate(population):
                logger.log(generation, i, chromosome.genes.raw, [chromosome.cost, *chromosome.cost_detail.values()])

        best_cost = population[0].cost
        stable_step = 0
        while stable_step < stable_step_for_exit:
            if strategy == "age":
                population = self._age_based_step(population, elitism, mutation_rate)
            elif strategy == "cost":
                population = self._cost_based_step(population, child_count, mutation_rate)
            else:
                raise ValueError("undefine survivor strategy")
            generation += 1

            print(generation, ">>>", [int(x.cost) for x in population])
            print(population[0].cost_detail)

            if is_logging:
                for i, chromosome in enumerate(population):
                    logger.log(generation, i, chromosome.genes.raw, [chromosome.cost, *chromosome.cost_detail.values()])

            if population[0].cost == best_cost:
                stable_step += 1
            elif population[0].cost < best_cost:
                best_cost = population[0].cost
                stable_step = 0
            else:
                print("Warning: cost increased??")

        return population[0]

    def _initialize(self, size):
        result = [Chromosome(self._gene_ruler.generate(), self._gene_ruler) for _ in range(size)]
        result.sort(key=attrgetter("cost"))

        return result

    def _cost_based_step(self, population, chlid_count, mutation_rate):
        result = population.copy()

        for _ in range(chlid_count // 2):
            result += self._reproduce_two_children(population, mutation_rate)

        result.sort(key=attrgetter("cost"))

        return result[:len(population)]

    def _age_based_step(self, population, elitism, mutation_rate):
        result = population[:elitism]

        while len(result) < len(population):
            result += self._reproduce_two_children(population, mutation_rate)

        result.sort(key=attrgetter("cost"))

        return result

    def _reproduce_two_children(self, population, mutation_rate):
        parent1 = choices(population, list(lerp(1, 0.5, len(population))))
        parent2 = choices(population, list(lerp(1, 0.5, len(population))))
        child1, child2 = parent1.crossover(parent2)

        if random.random() < mutation_rate:
            child1.mutate()

        if random.random() < mutation_rate:
            child2.mutate()

        return child1, child2


class GeneRuler:

    CLUSTER_COHESION = 8
    MARGINAL_PENALTY_FACTOR = 2

    def __init__(self, height, width, codes, mask=None, direction_masks=None):
        self._height = height
        self._width = width
        self._codes = codes

        if mask is None:
            self._target_mask = Grid(height=height, width=width, value=1)
        else:
            self._target_mask = Grid(mask)

        self._cluster_size = 1
        self._rules = []
        self._submasks = {}
        self._area_map = Grid(height=height, width=width, value=1)
        self._direction_masks = direction_masks

    @property
    def _cell_count(self):
        return self._target_mask.sum()

    def add_rule(self, rule):
        if isinstance(rule, ClusterCountMaxRule):
            self._cluster_size = self._cell_count // rule.maximum

        self._rules.append(rule)

    def add_submask(self, code, submask):
        self._submasks[code] = Grid(submask)

    def evaluate(self, genes):
        result = {}

        for rule in self._rules:
            result[str(rule)] = rule.evaluate(genes) ** GeneRuler.MARGINAL_PENALTY_FACTOR

        result["total"] = sum(result.values())

        return result

    def generate(self):
        grid = Grid(height=self._height, width=self._width, direction_masks=self._direction_masks)

        return self.fill(grid, self._target_mask, self._codes)

    def fill(self, genes, mask, codes=None):
        if codes is None:
            codes = self._codes

        target_coords = genes.get_coords(lambda x: mask[x] == 1)

        while True:
            if not target_coords:
                break

            r, c = randpop(target_coords)

            weights = [self._get_weight_at(genes, r, c, code) for code in codes]
            code = choices(codes, weights)
            genes, target_coords = self._fill_cluster(genes, target_coords, r, c, code, mask)

        return genes

    def _fill_cluster(self, grid, target_coords, r, c, code, mask):
        grid[r, c] = code
        current_cluster_size = 1

        neighbor_coords = []
        while True:
            neighbor_coords += grid.traverse_neighbor(r, c, lambda x: x, lambda x: grid[x] == 0
                                                                                   and mask[x] == 1
                                                                                   and self._target_mask[x] == 1
                                                                                   and x not in neighbor_coords)

            if not neighbor_coords:
                return grid, target_coords

            if current_cluster_size >= self._cluster_size:
                break

            neighbor_weights = [self._get_weight_at(grid, r, c, code) for r, c in neighbor_coords]

            if sum(neighbor_weights) == 0:
                break

            r, c = randpop(neighbor_coords, neighbor_weights)
            target_coords.remove((r, c))
            grid[r, c] = code

            current_cluster_size += 1

        # second phase

        while neighbor_coords:
            r, c = randpop(neighbor_coords)
            target_coords.remove((r, c))

            weights = [self._get_weight_at(grid, r, c, code) for code in self._codes]
            factored_weights = [weight * 1 if target_code == code else weight for weight, target_code in zip(weights, self._codes)]
            new_code = choices(self._codes, factored_weights)

            if code != new_code:
                code = new_code
                break

            grid[r, c] = code
            neighbor_coords += grid.traverse_neighbor(r, c, lambda x: x, lambda x: grid[x] == 0
                                                                                   and mask[x] == 1
                                                                                   and self._target_mask[x] == 1
                                                                                   and x not in neighbor_coords)

        return self._fill_cluster(grid, target_coords, r, c, code, mask)

    def _get_weight_at(self, grid, r, c, code):
        if code in self._submasks and self._submasks[code][r, c] == 0:
            return 0

        weight = self.CLUSTER_COHESION ** grid.count_neighbor(r, c, [code])

        for rule in self._rules:
            weight *= rule.weigh(grid, r, c, code)

        return weight


def main():
    pass
    # mask = [[1 if c > 4 and r < 105 else 0 for c in range(71)] for r in range(111)]
    # submask1 = [[1 if r < 40 else 0 for _ in range(71)] for r in range(111)]
    # submask2 = [[1 if c > 40 else 0 for c in range(71)] for _ in range(111)]
    # magnet1 = [[1 if 40 < c < 60 else 0 for c in range(71)] for _ in range(111)]
    # magnet2 = [[1 if c < 30 and r < 50 else 0 for c in range(71)] for r in range(111)]

    # up_mask = [[1 if r != 80 else 0 for c in range(71)] for r in range(111)]
    # down_mask = [[1 if r != 79 else 0 for c in range(71)] for r in range(111)]
    # left_mask = [[1 for c in range(71)] for r in range(111)]
    # right_mask = [[1 for c in range(71)] for r in range(111)]


    # ruler = GeneRuler(111, 71, list(range(1, 8)))

    # ruler.add_mask(mask)
    # ruler.add_cluster_rule(16, 8, 300)
    # ruler.add_area_rule([1000, 500, 1000, 250, 200, 1500, 1500], change_rate=0.5)

    # ruler.add_direction_masks({"up": up_mask, "down": down_mask, "left": left_mask, "right": right_mask})

    # ruler.add_submask(1, submask1)
    # ruler.add_submask(4, submask2)

    # ruler.add_magnet(2, magnet1, 4)
    # ruler.add_magnet(3, magnet1, 4)
    # ruler.add_magnet(3, magnet2, 4)
    # ruler.add_magnet(5, magnet2, 4)

    # ruler.add_repulsion_rule(6, 7)
    # ruler.add_attraction_rule(5, 7)
    # ruler.add_my_rule("a", lambda x: 15, 5, 10)







    # result, count = chromosome.genes.analyze_cluster()
    # print("the number of clusters", count)
    # # for code, dic in result.items():
    # #     print("code", code)
    # #     for name, arr in dic.items():
    # #         print(name, arr)

    # chromosome._evaluate()
    # print(chromosome.cost, chromosome._costs)

    # plot_grid(chromosome.genes)

    # chromosome.mutate()
    # print(chromosome.genes.analyze_cluster())
    # ruler.evaluate(chromosome.genes)
    # print(chromosome.cost, chromosome._costs)
    # plot_grid(chromosome.genes)


    # grid1 = ruler.generate()
    # grid2 = ruler.generate()

    # parent1 = Chromosome(grid1, ruler)
    # parent2 = Chromosome(grid2, ruler)
    # child1, child2 = parent1.crossover(parent2)

    # print(*[x.cost for x in [parent1, parent2, child1, child2]])
    # print(*[x._costs for x in [parent1, parent2, child1, child2]])

    # plot_grid(parent1.genes, parent2.genes, child1.genes, child2.genes)



    # ga = GeneticAlgorithm(ruler)
    # best = ga.run(size=20, elitism=2)
    # print(best.genes.analyze_cluster()[1])
    # print(best._costs)
    # plot_grid(best.genes)


if __name__ == "__main__":
    main()
