from mathutils import Grid


class MagnetRule:

    def __str__(self):
        return "_".join(map(str, ["magnet", self._gene, self._label]))

    def __init__(self, gene, mask, ideal, goal, label):
        self._gene = gene
        self._mask = Grid(mask)
        self._ideal = ideal
        self._goal = goal
        self._label = label

    def evaluate(self, genes):
        cluster_result, cluster_count = genes.analyze_cluster({self._gene: [self._mask]})

        result = 0

        for is_nears in cluster_result[self._gene]["is_near_mask"]:
            if not is_nears[0]:
                result += 1

        return (result - self._ideal) / (self._goal - self._ideal)

    def weigh(self, genes, r, c, gene):
        return 10 if gene == self._gene and self._mask.count_neighbor(r, c, [1]) > 0 else 1


class AreaMaxRule:

    def __str__(self):
        return "_".join(map(str, ["area_max", self._gene, self._maximum]))

    def __init__(self, gene, maximum, area_map):
        self._gene = gene
        self._area_map = area_map
        self._maximum = maximum

    @property
    def area_map(self):
        return self._area_map

    def evaluate(self, genes):
        return max(0, genes.each_sum(self._area_map)[self._gene] - self._maximum)

    def weigh(self, genes, r, c, gene):
        if gene != self._gene:
            return 1

        current_area = genes.each_sum(self._area_map)[self._gene]

        return (self._maximum - current_area) / self._maximum


class AreaMinRule:

    def __str__(self):
        return "_".join(map(str, ["area_min", self._gene, self._minimum]))

    def __init__(self, gene, minimum, area_map):
        self._gene = gene
        self._area_map = area_map
        self._minimum = minimum

    @property
    def area_map(self):
        return self._area_map

    def evaluate(self, genes):
        return max(0, self._minimum - genes.each_sum(self._area_map)[self._gene])

    def weigh(self, genes, r, c, gene):
        if gene != self._gene:
            return 1

        current_area = genes.each_sum(self._area_map)[self._gene]

        return 10 if current_area < self._minimum else 1


class ClusterSizeMinRule:

    def __str__(self):
        return "_".join(map(str, ["cluster_size_min", self._gene, self._minimum]))

    def __init__(self, gene, minimum):
        self._gene = gene
        self._minimum = minimum

    @property
    def minimum(self):
        return self._minimum

    def evaluate(self, genes):
        cluster_result, _ = genes.analyze_cluster()

        if self._gene in cluster_result:
            minimum = min(cluster_result[self._gene]["sizes"])
        else:
            minimum = 0

        return max(0, self._minimum - minimum)

    def weigh(self, genes, r, c, gene):
        return 1


class ClusterCountMaxRule:

    def __str__(self):
        return "cluster_count_max_" + str(self._maximum)

    def __init__(self, maximum):
        self._maximum = maximum

    @property
    def maximum(self):
        return self._maximum

    def evaluate(self, genes):
        _, cluster_count = genes.analyze_cluster()

        return max(0, cluster_count - self._maximum)

    def weigh(self, genes, r, c, gene):
        return 1


class AttractionRule:

    def __str__(self):
        return "_".join(map(str, ["attraction", self._from_gene, self._to_gene]))

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

    def weigh(self, genes, r, c, gene):
        if gene == self._from_gene and genes.count_neighbor(r, c, [self._to_gene]) > 0:
            return 10
        else:
            return 1


class RepulsionRule:

    def __str__(self):
        return "_".join(map(str, ["repulsion", self._gene1, self._gene2]))

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

    def weigh(self, genes, r, c, gene):
        if gene == self._gene1 and genes.count_neighbor(r, c, [self._gene2]) > 0:
            return 0
        elif gene == self._gene2 and genes.count_neighbor(r, c, [self._gene1]) > 0:
            return 0
        else:
            return 1
