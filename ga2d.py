import math
import random
from collections import defaultdict
from copy import deepcopy
from mathutils import lerp
from operator import attrgetter
from plotutils import plot_grid
from randutils import choices, randpop


class Grid:

    def __init__(self, *args, value=0, direction_masks=None):
        """
        `__init__(raw_data)`\n
        `__init__(height, width)`\n
        """
        self._direction_masks = direction_masks

        if len(args) == 1:
            self._raw_grid = args[0]
        else:
            self._raw_grid = [[value] * args[1] for _ in range(args[0])]

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
        return Grid(deepcopy(self._raw_grid), direction_masks=self._direction_masks)

    def get_coords(self, filter):
        return [(r, c) for r in range(self.height) for c in range(self.width) if filter((r, c))]

    def analyze_cluster(self):
        """
        ```
        return: (
            defaultdict {code: [size of each cluster]},
            the number of clusters
        )
        ```
        """
        result = defaultdict(lambda : dict(sizes=[], neighbors=[]))
        check_grid = self.copy()

        for r in range(self.height):
            for c in range(self.width):
                if check_grid[r, c] == 0:
                    continue

                code, count, neighbors = check_grid._fill_zeros_in_cluster(r, c, self)
                result[code]["sizes"].append(count)
                result[code]["neighbors"].append(neighbors)

        return result, sum(len(result[code]["sizes"]) for code in result)

    def _fill_zeros_in_cluster(self, r, c, original_map):
        target_code = self[r, c]
        target_coords = [(r, c)]
        count = 0
        neighbors = []

        while target_coords:
            r, c = target_coords.pop(0)
            neighbors += self.traverse_neighbor(r, c,
                                                lambda x: original_map[x],
                                                lambda x: original_map[x]
                                                          and original_map[x] != target_code)
            self[r, c] = 0
            count += 1

            target_coords += self.traverse_neighbor(r, c, lambda x: x, lambda x: self[x] == target_code and x not in target_coords)

        return target_code, count, list(set(neighbors))

    def count_neighbor(self, r, c, targets):
        return sum(self.traverse_neighbor(r, c, lambda x: 1, lambda x: self[x] in targets))

    def traverse_neighbor(self, r, c, action, filter=lambda: True):
        vectors = []

        if self._direction_masks is None:
            vectors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        else:
            if self._direction_masks["up"][r][c]:
                vectors.append((-1, 0))

            if self._direction_masks["down"][r][c]:
                vectors.append((1, 0))

            if self._direction_masks["left"][r][c]:
                vectors.append((0, -1))

            if self._direction_masks["right"][r][c]:
                vectors.append((0, 1))

        # vectors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        target_coords = [(r + dr, c + dc) for dr, dc in vectors]
        valid_coords = [(r, c) for r, c in target_coords if 0 <= r < self.height and 0 <= c < self.width]

        return [action((r, c)) for r, c in valid_coords if filter((r, c))]

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
        result = []

        for r in range(self.height):
            for c in range(self.width):
                if not filter((r, c)):
                    continue

                result.append(action((r, c)))

        return result

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

    def __init__(self, genes, gene_generator):
        self._genes = genes
        self._gene_generator = gene_generator
        self._cost = None

    @property
    def genes(self):
        return self._genes

    @property
    def cost(self):
        if self._cost is None:
            self._evaluate()
        return self._cost

    def crossover(self, partner):
        child_genes1 = self._genes.copy()
        child_genes2 = self._genes.copy()


        diff_coords = self._genes.get_diff_coords(partner._genes)


        for (gene1, gene2), coords in diff_coords.items():
            mask = Grid(self.genes.height, self.genes.width)
            for r, c in coords:
                mask[r, c] = 1
                child_genes1[r, c] = 0
                child_genes2[r, c] = 0

            self._gene_generator.fill(child_genes1, mask, [gene1, gene2])
            self._gene_generator.fill(child_genes2, mask, [gene1, gene2])

        return Chromosome(child_genes1, self._gene_generator), Chromosome(child_genes2, self._gene_generator)

    def _evaluate(self):
        self._cost, self._costs = self._gene_generator.evaluate(self._genes)

    def mutate(self):
        region_height = self.genes.height // 4
        region_width = self.genes.width // 4
        r_start = random.randint(0, self.genes.height - region_height)
        c_start = random.randint(0, self.genes.width - region_width)

        region_mask = Grid(self.genes.height, self.genes.width)

        for r in range(r_start, r_start + region_height):
            for c in range(c_start, c_start + region_width):
                region_mask[r, c] = 1
                self.genes[r, c] = 0

        self._gene_generator.fill(self.genes, region_mask)
        self._cost = None


class GeneticAlgorithm:

    def __init__(self, gene_generator):
        self._gene_generator = gene_generator

    def run(self, size=20, elitism=2, mutation_rate=0.05):
        population = self._initialize(size)
        generation = 1
        print(*(x.cost for x in population))

        while generation < 20:
            population = self._step(population, elitism, mutation_rate)
            generation += 1

            print(*(x.cost for x in population))
            print(population[0]._costs)
        return population[0]


    def _initialize(self, size):
        result = [Chromosome(self._gene_generator.generate(), self._gene_generator) for _ in range(size)]
        result.sort(key=attrgetter("cost"))

        return result

    def _step(self, population, elitism, mutation_rate):
        result = population[:elitism]

        while len(result) < len(population):
            parent1 = choices(population, list(lerp(1, 0.5, len(population) - elitism)))
            parent2 = choices(population, list(lerp(1, 0.5, len(population) - elitism)))
            child1, child2 = parent1.crossover(parent2)
            if random.random() < mutation_rate:
                child1.mutate()
            if random.random() < mutation_rate:
                child2.mutate()
            result.append(child1)
            result.append(child2)

        result.sort(key=attrgetter("cost"))

        return result


class GeneGenerator:

    def __init__(self, height, width, codes):
        self._height = height
        self._width = width
        self._codes = codes
        self._target_mask = Grid(height, width, value=1)
        self._cluster_size = 1
        self._cluster_cohesion = 1
        self._submasks = {}
        self._magnets = {}
        self._magnet_magnitudes = {}
        self._repulsions = defaultdict(list)
        self._attractions = defaultdict(list)
        self._area_map = Grid(height, width, value=1)
        self._max_areas = {}
        self._direction_masks = None

    def add_mask(self, mask):
        self._target_mask = Grid(mask)

    def add_cluster_rule(self, cluster_size, cluster_cohesion):
        self._cluster_size = cluster_size
        self._cluster_cohesion = cluster_cohesion

    def add_repulsion_rule(self, code1, code2):
        self._repulsions[code1].append(code2)
        self._repulsions[code2].append(code1)

    def add_attraction_rule(self, code1, code2):
        self._attractions[code1].append(code2)

    def change_area_map(self, area_map):
        self._area_map = Grid(area_map)

    def add_area_rule(self, max_areas):
        self._max_areas = {code: max_area for code, max_area in zip(self._codes, max_areas)}

    def add_submask(self, code, submask):
        self._submasks[code] = Grid(submask)

    def add_magnet(self, code, magnet, magnitude):
        self._magnets[code] = Grid(magnet, direction_masks=self._direction_masks)
        self._magnet_magnitudes[code] = magnitude

    def add_direction_masks(self, direction_masks):
        self._direction_masks = direction_masks

    def generate(self):
        grid = Grid(self._height, self._width, direction_masks=self._direction_masks)

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
        marginal_penalty_factor = 2

        # cluster size
        cluster_result, _ = genes.analyze_cluster()
        minimums = [min(cluster_result[code]["sizes"]) if cluster_result[code]["sizes"] else 0 for code in self._codes]
        raw_cluster_size_costs = [max(0, self._cluster_size - minimum) for minimum in minimums]
        cluster_size_cost = sum((cost / 1) ** marginal_penalty_factor  for cost in raw_cluster_size_costs)

        # magnet rule
        magnet_cost = 0
        for code in self._magnets:
            other_codes = self._codes.copy()
            other_codes.remove(code)
            raw_magnet_cost = genes.contain_or_adjacent(self._magnets[code], other_codes)
            mask_size = self._magnets[code].sum()
            mask_length = math.sqrt(mask_size)
            magnet_cost += ((raw_magnet_cost - mask_size) / (mask_size + mask_length * 3 - mask_size)) ** marginal_penalty_factor

        # area rule
        areas = self._area_map.masked_sum(genes)
        area_cost = 0
        for code in self._max_areas:
            raw_area_cost = max(0, areas[code] * 0.8 - self._max_areas[code])
            area_cost += (raw_area_cost / self._cluster_size) ** marginal_penalty_factor
            raw_area_cost = max(0, self._max_areas[code] - areas[code] * 1.2)
            area_cost += (raw_area_cost / self._cluster_size) ** marginal_penalty_factor

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
        raw_attraction_cost = 0
        for code, target_codes in self._attractions.items():
            for neighbors_of_a_cluster in cluster_result[code]["neighbors"]:
                for target_code in target_codes:
                    if target_code in neighbors_of_a_cluster:
                        continue

                    raw_attraction_cost += 1

        attraction_cost = ((raw_attraction_cost - 0) / (1 - 0)) ** marginal_penalty_factor

        cost = cluster_size_cost + magnet_cost + area_cost + repulsion_cost + attraction_cost

        return cost, {
            "cluster_size": cluster_size_cost,
            "magnet": magnet_cost,
            "area": area_cost,
            "repulsion": repulsion_cost,
            "attraction": attraction_cost
        }

    def _fill_cluster(self, grid, target_coords, accumulated_areas, r, c, code, mask):
        grid[r, c] = code
        accumulated_areas[code] += self._area_map[r, c]
        current_cluster_size = 1

        neighbor_coords = []
        while current_cluster_size < self._cluster_size:
            neighbor_coords += grid.traverse_neighbor(r, c, lambda x: x, lambda x: not grid[x]
                                                                                   and mask[x]
                                                                                   and self._target_mask[x]
                                                                                   and x not in neighbor_coords)

            if not neighbor_coords:
                return grid, target_coords, accumulated_areas

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
            weight *= self._magnet_magnitudes[code] ** self._magnets[code].count_neighbor(r, c, [1])

        if code in self._repulsions:
            weight *= 0 if grid.count_neighbor(r, c, self._repulsions[code]) else 1

        if code in self._attractions:
            weight *= 4 if grid.count_neighbor(r, c, self._attractions[code]) else 1

        if self._max_areas:
            weight *= (self._max_areas[code] * 1.2 - accumulated_areas[code]) / (self._max_areas[code] * 1.2)

        return weight


def main():
    mask = [[1 if c > 4 and r < 105 else 0 for c in range(71)] for r in range(111)]
    submask1 = [[1 if r < 40 else 0 for _ in range(71)] for r in range(111)]
    submask2 = [[1 if c > 40 else 0 for c in range(71)] for _ in range(111)]
    magnet1 = [[1 if 55 < c < 60 else 0 for c in range(71)] for _ in range(111)]
    magnet2 = [[1 if 20 < c < 30 and 40 < r < 50 else 0 for c in range(71)] for r in range(111)]

    up_mask = [[1 if r != 80 else 0 for c in range(71)] for r in range(111)]
    down_mask = [[1 if r != 79 else 0 for c in range(71)] for r in range(111)]
    left_mask = [[1 for c in range(71)] for r in range(111)]
    right_mask = [[1 for c in range(71)] for r in range(111)]




    generator = GeneGenerator(111, 71, list(range(1, 8)))

    generator.add_mask(mask)
    generator.add_cluster_rule(8, 8)
    generator.add_area_rule([1000, 500, 1000, 250, 1250, 1500, 1500])

    generator.add_direction_masks({"up": up_mask, "down": down_mask, "left": left_mask, "right": right_mask})

    generator.add_submask(1, submask1)
    generator.add_submask(4, submask2)

    generator.add_magnet(2, magnet1, 4)
    generator.add_magnet(3, magnet1, 4)
    generator.add_magnet(5, magnet2, 4)

    generator.add_repulsion_rule(6, 7)
    generator.add_attraction_rule(5, 7)


    genes = generator.generate()
    chromosome = Chromosome(genes, generator)

    result, count = chromosome.genes.analyze_cluster()
    print("the number of clusters", count)
    for code, dic in result.items():
        print("code", code)
        for name, arr in dic.items():
            print(name, arr)

    chromosome._evaluate()
    print(chromosome.cost, chromosome._costs)

    plot_grid(chromosome.genes)

    # chromosome.mutate()

    # print(chromosome.genes.analyze_cluster())
    # generator.evaluate(chromosome.genes)
    # print(chromosome.cost, chromosome._costs)
    # plot_grid(chromosome.genes)


    # grid1 = generator.generate()
    # grid2 = generator.generate()

    # parent1 = Chromosome(grid1, generator)
    # parent2 = Chromosome(grid2, generator)
    # child1, child2 = parent1.crossover(parent2)

    # print(*[x.cost for x in [parent1, parent2, child1, child2]])
    # print(*[x._costs for x in [parent1, parent2, child1, child2]])

    # plot_grid(parent1.genes, parent2.genes, child1.genes, child2.genes)



    # ga = GeneticAlgorithm(generator)
    # best = ga.run(size=20, elitism=2)
    # print(best.genes.analyze_cluster()[1])
    # print(best._costs)
    # plot_grid(best.genes)


if __name__ == "__main__":
    main()
