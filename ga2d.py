from galu import create_map
import random
from collections import defaultdict
from copy import deepcopy

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

def count_cluster(grid):
    result = 0
    check_grid = deepcopy(grid)

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if check_grid[r][c] == 0:
                continue

            result += 1
            check_grid = fill_zero_cluster(check_grid, r, c)

    return result

def fill_zero_cluster(grid, r, c):
    target_code = grid[r][c]
    target_coords = [(r, c)]

    while target_coords:
        r, c = target_coords.pop(0)
        grid[r][c] = 0

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if r + dr >= len(grid) or c + dc >= len(grid[0]) or r + dr < 0 or c + dc < 0:
                continue

            if grid[r + dr][c + dc] != target_code:
                continue

            target_coords.append((r + dr, c + dc))

    return grid

def choices_index(weights):
    normalized_weights = [weight / sum(weights) for weight in weights]
    accumulated_weights = [sum(normalized_weights[:i + 1]) for i in range(len(weights))]

    offset = random.random()
    for i, weight in enumerate(accumulated_weights):
        if offset <= weight:
            return i

def choices(sequence, weights=None):
    if weights is None:
        weights = [1] * len(sequence)
    return sequence[choices_index(weights)]

def randpop(sequence, weights=None):
    if weights is None:
        weights = [1] * len(sequence)
    return sequence.pop(choices_index(weights))

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
        self._target_mask = target_mask
        self._cluster_size = cluster_size

    @property
    def _width(self):
        return len(self._target_mask[0])

    @property
    def _height(self):
        return len(self._target_mask)

    def generate(self):
        result = self._create_empty_grid()

        while True:
            target_coords = self._list_unfilled_coords(result)

            if not target_coords:
                break

            r, c = choices(target_coords)
            weight = [1, 1, 1, 1]
            code = choices(self._codes, weight)
            result = self._fill_cluster(result, r, c, code)

        return result

    def _fill_cluster(self, grid, r, c, code):
        grid[r][c] = code
        current_cluster_size = 1

        neighbor_coords = []
        while current_cluster_size < self._cluster_size:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if r + dr >= self._height or c + dc >= self._width or r + dr < 0 or c + dc < 0:
                    continue

                if grid[r + dr][c + dc] == 0 and self._target_mask[r + dr][c + dc] == 1 and (r + dr, c + dc) not in neighbor_coords:
                    neighbor_coords.append((r + dr, c + dc))

            if not neighbor_coords:
                break

            neighbor_weights = [1] * len(neighbor_coords)
            r, c = randpop(neighbor_coords, neighbor_weights)
            grid[r][c] = code
            current_cluster_size += 1

        return grid

    def _list_unfilled_coords(self, grid):
        return [(r, c) for r in range(self._height)
                       for c in range(self._width)
                       if self._target_mask[r][c] == 1 and grid[r][c] == 0]

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
    genes = generator.generate()
    print(count_cluster(genes))
    plot(genes)


if __name__ == "__main__":
    main()
