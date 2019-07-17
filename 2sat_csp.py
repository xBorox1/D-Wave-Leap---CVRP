import dwavebinarycsp
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

# Number of variables.
n = int(input())

# Number of constraints.
m = int(input())

or_configurations = {(0, 1), (1, 0), (1, 1)}
csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)

# Only one of x and not x should be true.
for i in range(n):
    variables = [str(2 * i), str(2 * i + 1)]
    csp.add_constraint(lambda x, y : not (x and y), variables)

for i in range(m):
    a = int(input())
    b = int(input())
    variables = [str(a), str(b)]
    csp.add_constraint(or_configurations, variables)

# Converting to bqm.
sampler = EmbeddingComposite(DWaveSampler())
bqm = dwavebinarycsp.stitch(csp)
response = sampler.sample(bqm, num_reads=100)
sample = next(response.samples())

if csp.check(sample):
    print("Logical formula can be satisfied.")
else:
    print("Logical formula cannot be satisfied.")
