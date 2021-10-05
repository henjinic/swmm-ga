from matplotlib import pyplot as plt

from ga2d import Chromosome, GeneRuler, GeneticAlgorithm, Grid
from rules import AreaMaxRule, AreaMinRule, AttractionRule, ClusterCountMaxRule, ClusterSizeMinRule, MagnetRule, RepulsionRule


mask = Grid(height=50, width=50, value=1)
mask[0, 0] = mask[1, 0] = mask[49, 49] = 0

ruler = GeneRuler(50, 50, list(range(1, 9)), mask)
ruler.add_submask(5, [[1 if r > 25 and c > 25 else 0 for c in range(50)] for r in range(50)])
ruler.add_rule(AttractionRule(from_gene=1, to_gene=2, ideal=0, goal=1))
ruler.add_rule(RepulsionRule(gene1=3, gene2=4, ideal=0, goal=1))
ruler.add_rule(ClusterCountMaxRule(maximum=100))
ruler.add_rule(AreaMaxRule(gene=2, maximum=10, area_map=Grid(height=50, width=50, value=1)))
ruler.add_rule(AreaMinRule(gene=2, minimum=4, area_map=Grid(height=50, width=50, value=1)))
magnet = Grid(height=50, width=50, name="school")
for i in range(50):
    magnet[i, 30] = 1
ruler.add_rule(MagnetRule(gene=1, mask=magnet, ideal=0, goal=1))

parent1 = Chromosome(ruler.generate(), ruler)
parent2 = Chromosome(ruler.generate(), ruler)
print(parent1.cost_detail)
print(parent2.cost_detail)

child1, child2 = parent1.crossover(parent2)
print(child1.cost_detail)
print(child2.cost_detail)

plt.set_cmap("rainbow")
plt.subplot(221)
plt.imshow(parent1.genes.raw)
plt.subplot(222)
plt.imshow(parent2.genes.raw)
plt.subplot(223)
plt.imshow(child1.genes.raw)
plt.subplot(224)
plt.imshow(child2.genes.raw)
plt.show()
