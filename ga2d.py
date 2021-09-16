import random
from collections import defaultdict
from copy import deepcopy


class Chromosome:

    def __init__(self, genes):
        self._genes = genes

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


def main():
    parent1 = Chromosome([
        [0, 1, 3, 2],
        [3, 1, 2, 1],
        [2, 3, 2, 2],
        [2, 3, 1, 0]
    ])
    parent2 = Chromosome([
        [0, 2, 2, 3],
        [1, 2, 2, 3],
        [1, 1, 3, 3],
        [3, 3, 3, 0]
    ])
    child1, child2 = parent1.crossover(parent2)
    print(parent1)
    print(parent2)
    print(child1)
    print(child2)


if __name__ == "__main__":
    main()
