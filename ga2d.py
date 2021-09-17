import math
import random
from collections import defaultdict
from copy import deepcopy
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


def plot(*args):
    import matplotlib.pyplot as plt

    if len(args) == 1:
        fig, ax = plt.subplots()
        img = ax.imshow(args[0], cmap="gist_ncar")
        fig.colorbar(img, ax=ax, aspect=50)
    else:
        fig, axs = plt.subplots(1, len(args))

        for ax, grid in zip(axs, args):
            img = ax.imshow(grid, cmap="gist_ncar")
            fig.colorbar(img, ax=ax, aspect=50)

    plt.show()


class Chromosome:

    def __init__(self, genes):
        self._genes = genes

    @property
    def genes(self):
        return self._genes

    def crossover(self, partner):
        child_genes1 = deepcopy(self._genes)
        child_genes2 = deepcopy(self._genes)

        diff_coords = self._get_diff_coords(self._genes, partner._genes)

        for (gene1, gene2), coords in diff_coords.items():
            if random.randint(0, 1):
                coords.reverse()

            for r, c in coords[:len(coords) // 2]:
                child_genes1[r][c] = gene1
                child_genes2[r][c] = gene2

            for r, c in coords[len(coords) // 2:]:
                child_genes1[r][c] = gene2
                child_genes2[r][c] = gene1

        return Chromosome(child_genes1), Chromosome(child_genes2)

    def _get_diff_coords(self, genes1, genes2):
        result = defaultdict(list)

        for r in range(len(genes1)):
            for c in range(len(genes1[0])):
                if not genes1[r][c] or not genes2[r][c]:
                    continue

                if genes1[r][c] != genes2[r][c]:
                    result[frozenset((genes1[r][c], genes2[r][c]))].append((r, c))

        return result

    def __str__(self):
        lines = ["[" + " ".join(map(str, line)) + "]" for line in self._genes]
        aligned_lines = [" " + lines[i] if i else "[" + lines[i] for i in range(len(lines))]

        return "\n".join(aligned_lines) + "]"


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

    def add_mask(self, mask):
        self._target_mask = Grid(mask)

    def add_cluster_rule(self, cluster_size, cluster_cohesion):
        self._cluster_size = cluster_size
        self._cluster_cohesion = cluster_cohesion

    def add_repulsion_rule(self, code1, code2):
        self._repulsions[code1].append(code2)
        self._repulsions[code2].append(code1)

    def add_submask(self, code, submask):
        self._submasks[code] = Grid(submask)

    def add_magnet(self, code, magnet, magnitude):
        self._magnets[code] = Grid(magnet)
        self._magnet_magnitudes[code] = magnitude

    def generate(self):
        result = Grid(self._height, self._width)

        while True:
            target_coords = result.get_coords(lambda x: not result[x] and self._target_mask[x])

            if not target_coords:
                break

            r, c = choices(target_coords)

            weights = [self._get_weight_at(result, r, c, code) for code in self._codes]

            code = choices(self._codes, weights)
            result = self._fill_cluster(result, r, c, code)

        return result

    def _fill_cluster(self, grid, r, c, code):
        grid[r, c] = code
        current_cluster_size = 1

        neighbor_coords = []
        while current_cluster_size < self._cluster_size:
            neighbor_coords += grid.traverse_neighbor(r, c, lambda x: x, lambda x: not grid[x] and self._target_mask[x] and x not in neighbor_coords)

            if not neighbor_coords:
                break

            neighbor_weights = [self._get_weight_at(grid, r, c, code) for r, c in neighbor_coords]

            if not sum(neighbor_weights):
                break

            r, c = randpop(neighbor_coords, neighbor_weights)
            grid[r, c] = code
            current_cluster_size += 1

        return grid

    def _get_weight_at(self, grid, r, c, code):
        weight = self._cluster_cohesion ** grid.count_neighbor(r, c, [code])

        if code in self._submasks:
            weight *= self._submasks[code][r, c]

        if code in self._magnets:
            weight *= self._magnet_magnitudes[code] ** self._magnets[code].count_neighbor(r, c, [1])

        if code in self._repulsions:
            weight *= 0 if grid.count_neighbor(r, c, self._repulsions[code]) else 1

        return weight

def main():
    mask = [[1 if c > 4 and r < 105 else 0 for c in range(71)] for r in range(111)]
    submask1 = [[1 if r < 40 else 0 for c in range(71)] for r in range(111)]
    submask2 = [[1 if c > 30 else 0 for c in range(71)] for r in range(111)]
    magnet1 = [[1 if 55 < c < 60 else 0 for c in range(71)] for r in range(111)]
    magnet2 = [[1 if 20 < c < 30 and 40 < r < 50 else 0 for c in range(71)] for r in range(111)]

    generator = GeneGenerator(111, 71, list(range(1, 8)))
    generator.add_mask(mask)
    generator.add_cluster_rule(50, 4)

    generator.add_submask(1, submask1)
    generator.add_submask(4, submask2)

    generator.add_magnet(2, magnet1, 4)
    generator.add_magnet(3, magnet1, 4)
    generator.add_magnet(5, magnet2, 4)

    generator.add_repulsion_rule(7, 6)

    grid = generator.generate()
    print(grid.count_cluster())
    plot(grid._raw_grid)


    # grid1 = generator.generate()
    # grid2 = generator.generate()

    # parent1 = Chromosome(grid1._raw_grid)
    # parent2 = Chromosome(grid2._raw_grid)
    # child1, child2 = parent1.crossover(parent2)

    # print(*[Grid(x.genes).count_cluster() for x in [parent1, parent2, child1, child2]])
    # plot(parent1.genes, parent2.genes, child1.genes, child2.genes)



if __name__ == "__main__":
    main()
