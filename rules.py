# -*- coding: utf-8 -*-
from gridutils import analyze_cluster, count_neighbor, labeled_sum


class MagnetRule:

    def __str__(self):
        return "_".join(map(str, ["magnet", self._gene, self._label]))

    def __init__(self, gene, mask, ideal, goal, label):
        self._gene = gene
        self._mask = mask
        self._ideal = ideal
        self._goal = goal
        self._label = label

    def evaluate(self, genes):
        cluster_result = analyze_cluster(genes)
        result = sum(all(self._mask[r][c] == 0 for r, c in cluster["neighbor_coords"])
                     for cluster in cluster_result[self._gene])
        return (result - self._ideal) / (self._goal - self._ideal)

    def weigh(self, genes, r, c, gene):
        return 10 if gene == self._gene and count_neighbor(self._mask, r, c, [1]) > 0 else 1


class AreaMaxRule:

    def __str__(self):
        return "_".join(map(str, ["area_max", self._gene, f"{self._maximum:.4f}"]))

    def __init__(self, gene, maximum, area_map):
        self._gene = gene
        self._area_map = area_map
        self._maximum = maximum

    def evaluate(self, genes):
        return max(0, labeled_sum(genes, self._area_map)[self._gene] - self._maximum)

    def weigh(self, genes, r, c, gene, accumulated_areas):
        if gene != self._gene:
            return 1
        return (self._maximum - accumulated_areas[gene]) / self._maximum


class AreaMinRule:

    def __str__(self):
        return "_".join(map(str, ["area_min", self._gene, f"{self._minimum:.4f}"]))

    def __init__(self, gene, minimum, area_map):
        self._gene = gene
        self._area_map = area_map
        self._minimum = minimum

    def evaluate(self, genes):
        return max(0, self._minimum - labeled_sum(genes, self._area_map)[self._gene])

    def weigh(self, genes, r, c, gene, accumulated_areas):
        if gene != self._gene:
            return 1
        return 10 if accumulated_areas[gene] < self._minimum else 1


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
        cluster_result = analyze_cluster(genes)
        if self._gene in cluster_result:
            minimum = min(cluster["count"] for cluster in cluster_result[self._gene])
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
        cluster_result = analyze_cluster(genes)
        cluster_count = sum(len(clusters) for clusters in cluster_result.values())
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
        cluster_result = analyze_cluster(genes)
        cost = sum(all(genes[r][c] != self._to_gene for r, c in cluster["neighbor_coords"])
                   for cluster in cluster_result[self._from_gene])
        return (cost - self._ideal) / (self._goal - self._ideal)

    def weigh(self, genes, r, c, gene):
        if gene == self._from_gene and count_neighbor(genes, r, c, [self._to_gene]) > 0:
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
        for r, row in enumerate(genes):
            for c, e in enumerate(row):
                if e != self._gene1:
                    continue
                cost += count_neighbor(genes, r, c, [self._gene2])
        return (cost - self._ideal) / (self._goal - self._ideal)

    def weigh(self, genes, r, c, gene):
        if gene == self._gene1 and count_neighbor(genes, r, c, [self._gene2]) > 0:
            return 0
        elif gene == self._gene2 and count_neighbor(genes, r, c, [self._gene1]) > 0:
            return 0
        else:
            return 1
