from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
sampler = EmbeddingComposite(DWaveSampler())

# Number of variables.
n = int(input())

# Dict for qubo.
Q = dict()

# We have one variable for x and not x. 
for i in range(2 * n):
    for j in range(2 * n):
        Q[(i, j)] = 0

# Only one of x and not x should be 1.
# We want to minimize value of 2xy - x - y.
for i in range(n):
    Q[(2 * i, 2 * i)] = -1
    Q[(2 * i + 1, 2 * i + 1)] = -1
    Q[(2 * i, 2 * i + 1)] = 2

# Number of alternatives.
m = int(input())

# At least one of x and y should be true.
# We want to minimize value of xy - x - y.
for i in range(m):
    a = int(input())
    b = int(input())
    Q[(a, b)] += 1
    Q[(a, a)] -= 1
    Q[(b, b)] -= 1

response = sampler.sample_qubo(Q, num_reads=1000)

min_energy = response.first.energy
# Every satisfied constraint gives -1 energy.
if min_energy == -(n + m):
    print("Logical formula can be satisfied.")
else:
    print("Logical formula cannot be satisfied.")
