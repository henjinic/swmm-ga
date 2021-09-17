import random
from collections import defaultdict
from copy import deepcopy
from randutils import choices, randpop


class Grid:

    def __init__(self, *args):
        """
        `__init__(raw_data)`\n
        `__init__(height, width)`\n
        """
        if len(args) == 1:
            self._raw_grid = args[0]
        else:
            self._raw_grid = [[0] * args[1] for _ in range(args[0])]

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
        return {(r, c) for r in range(self.height)
                       for c in range(self.width)
                       if filter(self[r, c])}

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

            target_coords += self._traverse_neighbor(r, c, lambda x: x, lambda x: self[x] == target_code)

    def _traverse_neighbor(self, r, c, action, filter):
        result = []

        for dr, dc in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
            if r + dr >= self.height or c + dc >= self.width or r + dr < 0 or c + dc < 0:
                continue

            if not filter((r + dr, c + dc)):
                continue

            result.append(action((r + dr, c + dc)))

        return result


def plot(*args):
    import matplotlib.pyplot as plt

    if len(args) == 1:
        fig, ax = plt.subplots()
        img = ax.imshow(args[0])
        fig.colorbar(img, ax=ax, aspect=50)
    else:
        fig, axs = plt.subplots(1, len(args))

        for ax, grid in zip(axs, args):
            img = ax.imshow(grid)
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
                    result[frozenset([genes1[r][c], genes2[r][c]])].append((r, c))

        return result

    def __str__(self):
        lines = ["[" + " ".join(map(str, line)) + "]" for line in self._genes]
        aligned_lines = [" " + lines[i] if i else "[" + lines[i] for i in range(len(lines))]

        return "\n".join(aligned_lines) + "]"


class GeneGenerator:

    def __init__(self, codes, target_mask, cluster_size=10):
        self._codes = codes
        self._target_mask = Grid(target_mask)
        self._cluster_size = cluster_size

    @property
    def _width(self):
        return self._target_mask.width

    @property
    def _height(self):
        return self._target_mask.height

    def generate(self):
        result = Grid(self._target_mask.height, self._target_mask.width)

        while True:
            target_coords = list(result.get_coords(lambda x: not x) & self._target_mask.get_coords(lambda x: x))

            if not target_coords:
                break

            r, c = choices(target_coords)
            weight = [1, 1, 1, 1]
            code = choices(self._codes, weight)
            result = self._fill_cluster(result, r, c, code)

        return result

    def _fill_cluster(self, grid, r, c, code):
        grid[r, c] = code
        current_cluster_size = 1

        neighbor_coords = []
        while current_cluster_size < self._cluster_size:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if r + dr >= self._height or c + dc >= self._width or r + dr < 0 or c + dc < 0:
                    continue

                if grid[r + dr, c + dc] == 0 and self._target_mask[r + dr, c + dc] == 1 and (r + dr, c + dc) not in neighbor_coords:
                    neighbor_coords.append((r + dr, c + dc))

            if not neighbor_coords:
                break

            neighbor_weights = [1] * len(neighbor_coords)
            r, c = randpop(neighbor_coords, neighbor_weights)
            grid[r, c] = code
            current_cluster_size += 1

        return grid

    def _create_empty_grid(self):
        return [[0] * self._width for _ in range(self._height)]

def main():

    # parent1 = Chromosome([
    #     [0, 1, 3, 2],
    #     [3, 1, 2, 1],
    #     [2, 3, 2, 2],
    #     [2, 3, 1, 0]
    # ])
    # parent2 = Chromosome([
    #     [0, 2, 2, 3],
    #     [1, 2, 2, 3],
    #     [1, 1, 3, 3],
    #     [3, 3, 3, 0]
    # ])
    # child1, child2 = parent1.crossover(parent2)


    generator = GeneGenerator(
        codes=[1, 2, 3, 4],
        target_mask=[
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        ]
    )
    grid = generator.generate()
    print(grid.count_cluster())
    plot(grid._raw_grid)


if __name__ == "__main__":
    main()
