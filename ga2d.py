import random
from collections import defaultdict
from copy import deepcopy
from operator import attrgetter
from plotutils import plot
from randutils import choices, randpop


class Grid:

    def __init__(self, *args, value=0):
        """
        `__init__(raw_data)`\n
        `__init__(height, width)`\n
        """
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

    def copy(self):
        return Grid(deepcopy(self._raw_grid))

    def get_coords(self, filter):
        return [(r, c) for r in range(self.height) for c in range(self.width) if filter((r, c))]

    def count_cluster(self):
        result = 0
        check_grid = self.copy()

        for r in range(self.height):
            for c in range(self.width):
                if check_grid[r, c] == 0:
                    continue

                result += 1
                check_grid._fill_zeros_in_cluster(r, c)

        return result

    def _fill_zeros_in_cluster(self, r, c):
        target_code = self[r, c]
        target_coords = [(r, c)]

        while target_coords:
            r, c = target_coords.pop(0)
            self[r, c] = 0
            target_coords += self.traverse_neighbor(r, c, lambda x: x, lambda x: self[x] == target_code and x not in target_coords)

    def count_neighbor(self, r, c, targets):
        return sum(self.traverse_neighbor(r, c, lambda x: 1, lambda x: self[x] in targets))

    def traverse_neighbor(self, r, c, action, filter=lambda: True):
        target_coords = [(r + dr, c + dc) for dr, dc in [(-1, 0), (0, -1), (0, 1), (1, 0)]]
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

    def __str__(self):
        lines = ["[" + " ".join(map(str, line)) + "]" for line in self._raw_grid]
        aligned_lines = [" " + lines[i] if i else "[" + lines[i] for i in range(len(lines))]

        return "\n".join(aligned_lines) + "]"


class Chromosome:

    def __init__(self, genes):
        self._genes = genes
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
            if random.randint(0, 1):
                coords.reverse()

            for r, c in coords[:len(coords) // 2]:
                child_genes1[r, c] = gene1
                child_genes2[r, c] = gene2

            for r, c in coords[len(coords) // 2:]:
                child_genes1[r, c] = gene2
                child_genes2[r, c] = gene1

        return Chromosome(child_genes1), Chromosome(child_genes2)

    def _evaluate(self):
        self._cost = random.randint(10, 100)

    def mutate(self):
        pass


class GeneticAlgorithm:

    def __init__(self, gene_generator):
        self._gene_generator = gene_generator

    def run(self, size=20, elitism=2, mutation_rate=0.05):
        population = [Chromosome(self._gene_generator.generate()) for _ in range(size)]

        population.sort(key=attrgetter("cost"))
        generation = 1

        print(*(x.cost for x in population))

    def _step(self, population, elitism, mutation_rate):
        result = []

        mating_pool = self._select(population, len(population) - elitism)
        random.shuffle(mating_pool)

        for i in range(len(mating_pool) // 2):
            result.extend(mating_pool[i * 2].crossover(mating_pool[i * 2 + 1]))

        for chromosome in result:
            chromosome.mutate(mutation_rate)

        return result


    def _select(self, population, n, k=2):
        result = []

        while len(result) < n:
            result.append(random.sample(population, k).sort(key=attrgetter("cost"))[0])

        return result

def lerp(start, end, size):
    gap = (end - start) / (size - 1)

    yield start
    for i in range(1, size - 1):
        yield start + i * gap
    yield end


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
        self._area_map = Grid(height, width, value=1)
        self._max_areas = {}

    def add_mask(self, mask):
        self._target_mask = Grid(mask)

    def add_cluster_rule(self, cluster_size, cluster_cohesion):
        self._cluster_size = cluster_size
        self._cluster_cohesion = cluster_cohesion

    def add_repulsion_rule(self, code1, code2):
        self._repulsions[code1].append(code2)
        self._repulsions[code2].append(code1)

    def change_area_map(self, area_map):
        self._area_map = Grid(area_map)

    def add_area_rule(self, max_areas):
        self._max_areas = {code: max_area  for code, max_area in zip(self._codes, max_areas)}

    def add_submask(self, code, submask):
        self._submasks[code] = Grid(submask)

    def add_magnet(self, code, magnet, magnitude):
        self._magnets[code] = Grid(magnet)
        self._magnet_magnitudes[code] = magnitude

    def generate(self):
        grid = Grid(self._height, self._width)

        target_coords = grid.get_coords(lambda x: self._target_mask[x])
        accumulated_areas = defaultdict(int)

        while True:
            if not target_coords:
                break

            r, c = randpop(target_coords)

            weights = [self._get_weight_at(grid, r, c, code, accumulated_areas) for code in self._codes]
            code = choices(self._codes, weights)
            grid, target_coords, accumulated_areas = self._fill_cluster(grid, target_coords, accumulated_areas, r, c, code)

        return grid

    def _fill_cluster(self, grid, target_coords, accumulated_areas, r, c, code):
        grid[r, c] = code
        accumulated_areas[code] += self._area_map[r, c]
        current_cluster_size = 1

        neighbor_coords = []
        while current_cluster_size < self._cluster_size:
            neighbor_coords += grid.traverse_neighbor(r, c, lambda x: x, lambda x: not grid[x] and self._target_mask[x] and x not in neighbor_coords)

            if not neighbor_coords:
                break

            neighbor_weights = [self._get_weight_at(grid, r, c, code, accumulated_areas) for r, c in neighbor_coords]

            if not sum(neighbor_weights):
                break

            r, c = randpop(neighbor_coords, neighbor_weights)
            target_coords.remove((r, c))
            grid[r, c] = code
            accumulated_areas[code] += self._area_map[r, c]
            current_cluster_size += 1

        return grid, target_coords, accumulated_areas

    def _get_weight_at(self, grid, r, c, code, accumulated_areas):
        weight = self._cluster_cohesion ** grid.count_neighbor(r, c, [code])

        if code in self._submasks:
            weight *= self._submasks[code][r, c]

        if code in self._magnets:
            weight *= self._magnet_magnitudes[code] ** self._magnets[code].count_neighbor(r, c, [1])

        if code in self._repulsions:
            weight *= 0 if grid.count_neighbor(r, c, self._repulsions[code]) else 1

        if self._max_areas:
            weight *= (self._max_areas[code] - accumulated_areas[code]) / self._max_areas[code]

        return weight


def main():
    mask = [[1 if c > 4 and r < 105 else 0 for c in range(71)] for r in range(111)]
    submask1 = [[1 if r < 40 else 0 for _ in range(71)] for r in range(111)]
    submask2 = [[1 if c > 40 else 0 for c in range(71)] for _ in range(111)]
    magnet1 = [[1 if 55 < c < 60 else 0 for c in range(71)] for _ in range(111)]
    magnet2 = [[1 if 20 < c < 30 and 40 < r < 50 else 0 for c in range(71)] for r in range(111)]

    generator = GeneGenerator(111, 71, list(range(1, 8)))

    generator.add_mask(mask)
    generator.add_cluster_rule(30, 10)
    generator.add_area_rule([1000, 500, 1000, 1250, 250, 1500, 1500])

    generator.add_submask(1, submask1)
    generator.add_submask(4, submask2)

    generator.add_magnet(2, magnet1, 10)
    generator.add_magnet(3, magnet1, 10)
    generator.add_magnet(5, magnet2, 10)

    generator.add_repulsion_rule(7, 6)



    # genes = generator.generate()
    # print(genes.count_cluster())
    # plot(genes)


    # grid1 = generator.generate()
    # grid2 = generator.generate()

    # parent1 = Chromosome(grid1)
    # parent2 = Chromosome(grid2)
    # child1, child2 = parent1.crossover(parent2)

    # print(*[x.genes.count_cluster() for x in [parent1, parent2, child1, child2]])
    # plot(parent1.genes, parent2.genes, child1.genes, child2.genes)

    ga = GeneticAlgorithm(generator)
    ga.run(3)



if __name__ == "__main__":
    main()
