from matplotlib import pyplot as plt

from ga2d import Chromosome, ClusterSizeMinRule, GeneRuler, GeneticAlgorithm, Grid
from ga2d import AreaMaxRule, AreaMinRule, AttractionRule, ClusterCountMaxRule, ClusterSizeMinRule, MagnetRule, RepulsionRule


ruler = GeneRuler(10, 10, list(range(1, 7)))
ruler.add_cluster_rule(2, 4, 300)
ruler.add_rule(AttractionRule(from_gene=1, to_gene=2, ideal=0, goal=1))
ruler.add_rule(RepulsionRule(gene1=3, gene2=4, ideal=0, goal=1))
ruler.add_rule(ClusterCountMaxRule(maximum=30))
ruler.add_rule(ClusterSizeMinRule(gene=1, minimum=3))
ruler.add_rule(AreaMaxRule(gene=2, maximum=10, area_map=Grid(height=10, width=10, value=1)))
ruler.add_rule(AreaMinRule(gene=2, minimum=4, area_map=Grid(height=10, width=10, value=1)))
magnet = Grid(height=10, width=10, name="school")
magnet[1, 1] = 1
magnet[2, 1] = 1
magnet[3, 1] = 1
ruler.add_rule(MagnetRule(gene=1, mask=magnet, ideal=0, goal=1))

parent1 = Chromosome(ruler.generate(), ruler)
parent2 = Chromosome(ruler.generate(), ruler)
print(parent1.cost_detail)
print(parent2.cost_detail)

child1, child2 = parent1.crossover(parent2)
print(child1.cost_detail)
print(child2.cost_detail)

plt.subplot(221)
plt.imshow(parent1.genes.raw)
plt.subplot(222)
plt.imshow(parent2.genes.raw)
plt.subplot(223)
plt.imshow(child1.genes.raw)
plt.subplot(224)
plt.imshow(child2.genes.raw)
plt.show()
