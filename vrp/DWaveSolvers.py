from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_qbsolv import QBSolv
from hybrid.reference.kerberos import KerberosSampler
from dimod.reference.samplers import ExactSolver
import hybrid
import dimod
import neal

# Creates hybrid solver.
def hybrid_solver():
    workflow = hybrid.Loop(
        hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        hybrid.EnergyImpactDecomposer(size=30, rolling=True, rolling_history=0.75)
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()) | hybrid.ArgMin(), convergence=3)
    return hybrid.HybridSampler(workflow)

def get_solver(solver_type):
    solver = None
    if solver_type == 'exact':
        solver = ExactSolver()
    if solver_type == 'standard':
        solver = EmbeddingComposite(DWaveSolver())
    if solver_type == 'hybrid':
        solver = hybrid_solver()
    if solver_type == 'kerberos':
        solver = KerberosSampler()
    if solver_type == 'qbsolv':
        solver = QBSolv()
    return solver

# Solves qubo on qpu.
def solve_qubo(qubo, solver_type = 'qbsolv', limit = 1, num_repeats = 50):
    sampler = get_solver(solver_type)
    response = sampler.sample_qubo(qubo.dict, num_repeats = num_repeats)
    return list(response)[:limit]
    
