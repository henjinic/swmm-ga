# -*- coding: utf-8 -*-
import random
from operator import attrgetter

from gridutils import grid_sum, labeled_sum
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

    def __init__(self, height, width, codes, mask=None, direction_masks=None, area_map=None):
        self._height = height
        self._width = width
        self._codes = codes

        if mask is None:
            self._target_mask = Grid(height=height, width=width, value=1)
        else:
            self._target_mask = Grid(mask)

        self._cluster_size = 1
        self._rules = []
        self._area_rules = []
        self._submasks = {}
        self._direction_masks = direction_masks

        if area_map is None:
            self._area_map = [[1] * width for _ in range(height)]
        else:
            self._area_map = area_map


    @property
    def _cell_count(self):
        return grid_sum(self._target_mask.raw)
        # return self._target_mask.sum()

    def add_rule(self, rule):
        if isinstance(rule, ClusterCountMaxRule):
            self._cluster_size = self._cell_count // (rule.maximum * 0.75)

        self._rules.append(rule)

    def add_area_rule(self, rule):
        self._area_rules.append(rule)

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

        # target_coords = genes.get_coords(lambda r, c: mask[r, c] == 1)
        target_coords = [(r, c) for r in range(len(mask.raw))
                                for c in range(len(mask.raw[0]))
                                if mask.raw[r][c] != 0]
        # accumulated_areas = genes.each_sum(self._area_map)
        accumulated_areas = labeled_sum(genes.raw, self._area_map)

        while True:
            if not target_coords:
                break

            r, c = randpop(target_coords)

            weights = [self._get_weight_at(genes, r, c, code, accumulated_areas) for code in codes]
            code = choices(codes, weights)
            genes, target_coords, accumulated_areas = self._fill_cluster(genes, target_coords, accumulated_areas, r, c, code, mask)

        return genes

    def _fill_cluster(self, grid, target_coords, accumulated_areas, r, c, code, mask):
        grid[r, c] = code
        accumulated_areas[code] += self._area_map[r][c]
        current_cluster_size = 1

        neighbor_coords = []
        while True:
            neighbor_coords += grid.traverse_neighbor(r, c, lambda r, c: (r, c),
                                                            lambda r, c: grid[r, c] == 0
                                                                         and mask[r, c] == 1
                                                                         and self._target_mask[r, c] == 1
                                                                         and (r, c) not in neighbor_coords)

            if not neighbor_coords:
                return grid, target_coords, accumulated_areas

            if current_cluster_size >= self._cluster_size:
                break

            neighbor_weights = [self._get_weight_at(grid, r, c, code, accumulated_areas) for r, c in neighbor_coords]

            if sum(neighbor_weights) == 0:
                break

            r, c = randpop(neighbor_coords, neighbor_weights)
            target_coords.remove((r, c))
            grid[r, c] = code
            accumulated_areas[code] += self._area_map[r][c]
            current_cluster_size += 1

        # second phase

        while neighbor_coords:
            r, c = randpop(neighbor_coords)
            target_coords.remove((r, c))

            weights = [self._get_weight_at(grid, r, c, code, accumulated_areas) for code in self._codes]
            factored_weights = [weight * 1 if target_code == code else weight for weight, target_code in zip(weights, self._codes)]
            new_code = choices(self._codes, factored_weights)

            if code != new_code:
                code = new_code
                break

            grid[r, c] = code
            accumulated_areas[code] += self._area_map[r][c]
            neighbor_coords += grid.traverse_neighbor(r, c, lambda r, c: (r, c),
                                                            lambda r, c: grid[r, c] == 0
                                                                         and mask[r, c] == 1
                                                                         and self._target_mask[r, c] == 1
                                                                         and (r, c) not in neighbor_coords)

        return self._fill_cluster(grid, target_coords, accumulated_areas, r, c, code, mask)

    def _get_weight_at(self, grid, r, c, code, accumulated_areas):
        if code in self._submasks and self._submasks[code][r, c] == 0:
            return 0

        weight = self.CLUSTER_COHESION ** grid.count_neighbor(r, c, [code])

        for rule in self._rules:
            weight *= rule.weigh(grid, r, c, code)

        for rule in self._area_rules:
            weight *= rule.weigh(grid, r, c, code, accumulated_areas)

        return weight


def main():
    pass


if __name__ == "__main__":
    main()
