from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_qbsolv import QBSolv
import hybrid

# Creates hybrid solver.
def hybrid_solver():
    workflow = hybrid.Loop(
        hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        hybrid.EnergyImpactDecomposer(size=30, rolling=True, rolling_history=0.75)
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()) | hybrid.ArgMin(), convergence=3)
    return hybrid.HybridSampler(workflow)

# Solves qubo on cpu.
def solve_qubo_on_cpu(qubo, limit = 1, num_repeats = 50):
    response = QBSolv().sample_qubo(qubo.dict, num_repeats = num_repeats)
    return list(response.samples())[:limit]

# Solves qubo on qpu.
def solve_qubo_on_qpu(qubo, limit = 1):
    result = hybrid_solver().sample_qubo(qubo.dict)
    return list(result)[:limit]
    
