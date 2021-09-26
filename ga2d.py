import random
from collections import defaultdict
from copy import deepcopy
from operator import attrgetter

from logger import GALogger27
from mathutils import lerp
from randutils import choices, randpop

# 3
from logger import GALogger
from plotutils import plot_grid
#


class Grid:

    def __init__(self, raw_data=None, height=None, width=None, value=0, direction_masks=None):
        """
        `__init__(raw_data)`\n
        `__init__(height, width, value=0)`\n
        """

        if raw_data is not None:
            self._raw_grid = raw_data
        else:
            self._raw_grid = [[value] * width for _ in range(height)]

        self._direction_masks = direction_masks

        if direction_masks is not None:
            self._unit_vector_cache = defaultdict(list)

            for r in range(self.height):
                for c in range(self.width):

                    if self._direction_masks["up"][r][c]:
                        self._unit_vector_cache[r, c].append((-1, 0))

                    if self._direction_masks["down"][r][c]:
                        self._unit_vector_cache[r, c].append((1, 0))

                    if self._direction_masks["left"][r][c]:
                        self._unit_vector_cache[r, c].append((0, -1))

                    if self._direction_masks["right"][r][c]:
                        self._unit_vector_cache[r, c].append((0, 1))

    def __getitem__(self, coord):
        return self._raw_grid[coord[0]][coord[1]]

    def __setitem__(self, coord, value):
        self._raw_grid[coord[0]][coord[1]] = value

    @property
    def raw(self):
        return self._raw_grid

    @property
    def height(self):
        return len(self._raw_grid)

    @property
    def width(self):
        return len(self._raw_grid[0])

    @property
    def direction_masks(self):
        return self._direction_masks

    def copy(self):
        return Grid(raw_data=deepcopy(self._raw_grid), direction_masks=self._direction_masks)

    def get_coords(self, filter):
        return [(r, c) for r in range(self.height) for c in range(self.width) if filter((r, c))]

    def analyze_cluster(self, masks=None):
        """
        ```
        return: (
            defaultdict {code: [size of each cluster]},
            the number of clusters
        )
        ```
        """
        if masks is None:
            masks = {}
            result = defaultdict(lambda : dict(sizes=[], neighbors=[]))
        else:
            result = defaultdict(lambda : dict(sizes=[], neighbors=[], is_near_mask=[]))


        check_grid = self.copy()

        for r in range(self.height):
            for c in range(self.width):
                if check_grid[r, c] == 0:
                    continue

                code, count, neighbors, is_near_mask = check_grid._fill_zeros_in_cluster(r, c, self, masks)
                result[code]["sizes"].append(count)
                result[code]["neighbors"].append(neighbors)
                if code in masks:
                    result[code]["is_near_mask"].append(is_near_mask)

        return result, sum(len(result[code]["sizes"]) for code in result)

    def _fill_zeros_in_cluster(self, r, c, original_map, masks):
        target_code = self[r, c]
        target_coords = [(r, c)]
        count = 0
        neighbors = []
        if target_code in masks:
            is_near_mask = [False] * len(masks[target_code])
        else:
            is_near_mask = None

        while target_coords:
            r, c = target_coords.pop(0)
            neighbors += self.traverse_neighbor(r, c,
                                                lambda x: original_map[x],
                                                lambda x: original_map[x]
                                                          and original_map[x] != target_code)

            if target_code in masks:
                for i, mask in enumerate(masks[target_code]):
                    if is_near_mask[i]:
                        continue

                    is_near_mask[i] = bool(mask.count_neighbor(r, c, [1]))

            self[r, c] = 0
            count += 1

            target_coords += self.traverse_neighbor(r, c, lambda x: x, lambda x: self[x] == target_code and x not in target_coords)

        return target_code, count, list(set(neighbors)), is_near_mask

    def count_neighbor(self, r, c, targets):
        return sum(self.traverse_neighbor(r, c, lambda x: 1, lambda x: self[x] in targets))

    def traverse_neighbor(self, r, c, action, filter=lambda: True):
        if self._direction_masks is None:
            vectors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        else:
            vectors = self._unit_vector_cache[r, c]

        return [action((r + dr, c + dc)) for dr, dc in vectors if 0 <= r + dr < self.height and 0 <= c + dc < self.width and filter((r + dr, c + dc))]

    def get_diff_coords(self, partner):
        """
        ```
        # example
        arg1: [[1, 2], | arg2: [[1, 1],
               [1, 1]] |        [2, 3]]

        return: defaultdict {
            frozenset {1, 2}: [(0, 1), (1, 0)]
            frozenset {1, 3}: [(1, 1)]
        }
        ```
        """
        result = defaultdict(list)

        for r in range(self.height):
            for c in range(self.width):
                if self[r, c] != partner[r, c]:
                    result[frozenset((self[r, c], partner[r, c]))].append((r, c))

        return result

    def contain_or_adjacent(self, mask, targets):
        return sum(self.traverse(lambda x: 1, lambda x: (mask[x] or mask.count_neighbor(*x, [1])) and self[x] in targets))

    def sum(self):
        return sum(self.traverse(lambda x: self[x], lambda x: True))

    def traverse(self, action, filter):
        return [action((r, c)) for r in range(self.height) for c in range(self.width) if filter((r, c))]

    def masked_sum(self, mask):
        result = defaultdict(int)

        for r in range(self.height):
            for c in range(self.width):
                if not mask[r, c]:
                    continue

                result[mask[r, c]] += self[r, c]

        return result


    def __str__(self):
        lines = ["[" + " ".join(map(str, line)) + "]" for line in self._raw_grid]
        aligned_lines = [" " + lines[i] if i else "[" + lines[i] for i in range(len(lines))]

        return "\n".join(aligned_lines) + "]"


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
        if self._costs is None:
            self._evaluate()
        return self._costs["total"]

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

    def _evaluate(self):
        self._costs = self._gene_ruler.evaluate(self._genes)

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

    def __init__(self, height, width, codes):
        self._height = height
        self._width = width
        self._codes = codes
        self._target_mask = Grid(height=height, width=width, value=1)
        self._cluster_size = 1
        self._cluster_cohesion = 1
        self._cluster_count = None
        self._submasks = {}
        self._magnets = defaultdict(list)
        self._magnet_magnitudes = defaultdict(list)
        self._repulsions = defaultdict(list)
        self._attractions = defaultdict(list)
        self._area_map = Grid(height=height, width=width, value=1)
        self._max_areas = {}
        self._area_change_rate = None
        self._area_change_amount = None
        self._direction_masks = None
        self._my_rules = {}
        self._rules = []

    def add_rule(self, rule):
        self._rules.append(rule)

    def add_mask(self, mask):
        self._target_mask = Grid(mask)

    def add_cluster_rule(self, cluster_size, cluster_cohesion, cluster_count):
        self._cluster_size = cluster_size
        self._cluster_cohesion = cluster_cohesion
        self._cluster_count = cluster_count

    def add_repulsion_rule(self, code1, code2):
        self._repulsions[code1].append(code2)
        self._repulsions[code2].append(code1)

    def add_attraction_rule(self, code1, code2):
        self._attractions[code1].append(code2)

    def change_area_map(self, area_map):
        self._area_map = Grid(area_map)

    def add_area_rule(self, max_areas, change_rate=None, change_amount=None):
        self._max_areas = {code: max_area for code, max_area in zip(self._codes, max_areas)}
        self._area_change_rate = change_rate
        self._area_change_amount = change_amount

    def add_submask(self, code, submask):
        self._submasks[code] = Grid(submask)

    def add_magnet(self, code, magnet, magnitude):
        self._magnets[code].append(Grid(magnet, direction_masks=self._direction_masks))
        self._magnet_magnitudes[code].append(magnitude)

    def add_direction_masks(self, direction_masks):
        self._direction_masks = direction_masks

    def add_my_rule(self, name, cost_function, ideal, goal):
        self._my_rules[name] = {}
        self._my_rules[name]["cost_function"] = cost_function
        self._my_rules[name]["ideal"] = ideal
        self._my_rules[name]["goal"] = goal

    def generate(self):
        grid = Grid(height=self._height, width=self._width, direction_masks=self._direction_masks)

        target_coords = grid.get_coords(lambda x: self._target_mask[x])
        accumulated_areas = defaultdict(int)

        while True:
            if not target_coords:
                break

            r, c = randpop(target_coords)

            weights = [self._get_weight_at(grid, r, c, code, accumulated_areas) for code in self._codes]

            code = choices(self._codes, weights)
            grid, target_coords, accumulated_areas = self._fill_cluster(grid, target_coords, accumulated_areas, r, c, code, self._target_mask)

        return grid

    def fill(self, genes, mask, codes=None):
        if codes is None:
            codes = self._codes

        target_coords = genes.get_coords(lambda x: mask[x] and self._target_mask[x])
        accumulated_areas = self._area_map.masked_sum(genes)

        while True:
            if not target_coords:
                break

            r, c = randpop(target_coords)

            weights = [self._get_weight_at(genes, r, c, code, accumulated_areas) for code in codes]
            code = choices(codes, weights)
            genes, target_coords, accumulated_areas = self._fill_cluster(genes, target_coords, accumulated_areas, r, c, code, mask)

        return genes

    def evaluate(self, genes):
        result = {}

        marginal_penalty_factor = 1

        cluster_result, cluster_count = genes.analyze_cluster(self._magnets)

        # # cluster size
        # minimums = [min(cluster_result[code]["sizes"]) if cluster_result[code]["sizes"] else 0 for code in self._codes]
        # raw_cluster_size_costs = [max(0, self._cluster_size - minimum) for minimum in minimums]
        # cluster_size_cost = sum((cost / 1) ** marginal_penalty_factor  for cost in raw_cluster_size_costs)

        # cluster count
        cluster_count_cost = 0
        if self._cluster_count is not None:
            cluster_count_cost = max(0, cluster_count - self._cluster_count) ** marginal_penalty_factor

        # magnet rule

        raw_magnet_cost = 0
        for code in self._magnets:
            for is_nears in cluster_result[code]["is_near_mask"]:
                for is_near, magnitude in zip(is_nears, self._magnet_magnitudes[code]):
                    if not is_near:
                        raw_magnet_cost += magnitude / 4

            # raw_magnet_cost += sum([is_nears.count(False) for is_nears in cluster_result[code]["is_near_mask"]])
        magnet_cost = ((raw_magnet_cost - 0) / (1 - 0)) ** marginal_penalty_factor

        # area rule
        areas = self._area_map.masked_sum(genes)
        area_cost = 0
        for code in self._max_areas:
            if self._area_change_rate is not None:
                raw_area_cost = max(0, self._max_areas[code] * (1 - self._area_change_rate) - areas[code])
                area_cost += (raw_area_cost / 1) ** marginal_penalty_factor
                raw_area_cost = max(0, areas[code] - self._max_areas[code] * (1 + self._area_change_rate))
                area_cost += (raw_area_cost / 1) ** marginal_penalty_factor
            else:
                raw_area_cost = max(0, (self._max_areas[code] - self._area_change_amount) - areas[code])
                area_cost += (raw_area_cost / 1) ** marginal_penalty_factor
                raw_area_cost = max(0, areas[code] - (self._max_areas[code] + self._area_change_amount))
                area_cost += (raw_area_cost / 1) ** marginal_penalty_factor

        # repulsion rule
        raw_repulsion_cost = 0
        for r in range(genes.height):
            for c in range(genes.width):
                if genes[r, c] not in self._repulsions:
                    continue

                raw_repulsion_cost += genes.count_neighbor(r, c, self._repulsions[genes[r, c]])

        raw_repulsion_cost /= 2
        repulsion_cost = ((raw_repulsion_cost - 0) / (1 - 0)) ** marginal_penalty_factor

        # attraction rule
        # raw_attraction_cost = 0
        # for code, target_codes in self._attractions.items():
        #     for neighbors_of_a_cluster in cluster_result[code]["neighbors"]:
        #         for target_code in target_codes:
        #             if target_code in neighbors_of_a_cluster:
        #                 continue

        #             raw_attraction_cost += 1

        # attraction_cost = ((raw_attraction_cost - 0) / (1 - 0)) ** marginal_penalty_factor

        cost = 0
        for rule in self._rules:
            result[str(rule)] = rule.evaluate(genes) ** marginal_penalty_factor

        my_costs = {}
        for my_rule_name, rule_data in self._my_rules.items():
            raw_cost = rule_data["cost_function"](genes.raw)

            my_costs[my_rule_name] = ((raw_cost - rule_data["ideal"]) / (rule_data["goal"] - rule_data["ideal"])) ** marginal_penalty_factor


        # cost += cluster_count_cost + magnet_cost + area_cost + repulsion_cost + attraction_cost + sum(my_costs.values())

        return result

        # return {
        #     "total": cost,
        #     "cluster_count": cluster_count_cost,
        #     "magnet": magnet_cost,
        #     "area": area_cost,
        #     "repulsion": repulsion_cost,
        #     "attraction": attraction_cost,
        #     **my_costs
        # }

    def _fill_cluster(self, grid, target_coords, accumulated_areas, r, c, code, mask):
        grid[r, c] = code
        accumulated_areas[code] += self._area_map[r, c]
        current_cluster_size = 1

        neighbor_coords = []
        while True:
            neighbor_coords += grid.traverse_neighbor(r, c, lambda x: x, lambda x: not grid[x]
                                                                                   and mask[x]
                                                                                   and self._target_mask[x]
                                                                                   and x not in neighbor_coords)

            if not neighbor_coords:
                return grid, target_coords, accumulated_areas

            if current_cluster_size >= self._cluster_size:
                break

            neighbor_weights = [self._get_weight_at(grid, r, c, code, accumulated_areas) for r, c in neighbor_coords]

            if not sum(neighbor_weights):
                break

            r, c = randpop(neighbor_coords, neighbor_weights)
            target_coords.remove((r, c))
            grid[r, c] = code

            accumulated_areas[code] += self._area_map[r, c]
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
            accumulated_areas[code] += self._area_map[r, c]
            neighbor_coords += grid.traverse_neighbor(r, c, lambda x: x, lambda x: not grid[x]
                                                                                   and mask[x]
                                                                                   and self._target_mask[x]
                                                                                   and x not in neighbor_coords)

        return self._fill_cluster(grid, target_coords, accumulated_areas, r, c, code, mask)

        # return grid, target_coords, accumulated_areas

    def _get_weight_at(self, grid, r, c, code, accumulated_areas):
        weight = self._cluster_cohesion ** grid.count_neighbor(r, c, [code])

        if code in self._submasks:
            weight *= self._submasks[code][r, c]

        if code in self._magnets:
            for magnet, magnitude in zip(self._magnets[code], self._magnet_magnitudes):
                weight *= magnitude ** magnet.count_neighbor(r, c, [1])

        if code in self._repulsions:
            weight *= 0 if grid.count_neighbor(r, c, self._repulsions[code]) else 1

        if code in self._attractions:
            weight *= 4 if grid.count_neighbor(r, c, self._attractions[code]) else 1

        if self._max_areas:
            if self._area_change_rate is not None:
                weight *= (self._max_areas[code] * (1 + self._area_change_rate) - accumulated_areas[code]) / (self._max_areas[code] * (1 + self._area_change_rate))
            else:
                weight *= (self._max_areas[code] + self._area_change_amount - accumulated_areas[code]) / (self._max_areas[code] + self._area_change_amount)
        return weight


class AttractionRule:

    def __str__(self):
        return "attraction_" + str(self._from_gene) + "_" + str(self._to_gene)

    def __init__(self, from_gene, to_gene, ideal, goal):
        self._from_gene = from_gene
        self._to_gene = to_gene
        self._ideal = ideal
        self._goal = goal

    def evaluate(self, genes):
        cost = 0
        cluster_result, _ = genes.analyze_cluster()

        for neighbors_of_a_cluster in cluster_result[self._from_gene]["neighbors"]:
            if self._to_gene in neighbors_of_a_cluster:
                continue

            cost += 1

        return (cost - self._ideal) / (self._goal - self._ideal)


class RepulsionRule:

    def __str__(self):
        return "repulsion_" + str(self._gene1) + "_" + str(self._gene2)

    def __init__(self, gene1, gene2, ideal, goal):
        self._gene1 = gene1
        self._gene2 = gene2
        self._ideal = ideal
        self._goal = goal

    def evaluate(self, genes):
        cost = 0
        for r in range(genes.height):
            for c in range(genes.width):
                if genes[r, c] != self._gene1:
                    continue

                cost += genes.count_neighbor(r, c, [self._gene2])

        return (cost - self._ideal) / (self._goal - self._ideal)



def main():
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



    ruler = GeneRuler(10, 10, list(range(1, 5)))
    ruler.add_cluster_rule(2, 8, 300)
    ruler.add_rule(AttractionRule(from_gene=1, to_gene=2, ideal=0, goal=1))
    ruler.add_rule(RepulsionRule(gene1=3, gene2=4, ideal=0, goal=1))
    genes = ruler.generate()
    chromosome = Chromosome(genes, ruler)
    print(chromosome.cost_detail)
    plot_grid(chromosome.genes)



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
