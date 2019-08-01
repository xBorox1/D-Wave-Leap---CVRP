from qubo_helper import Qubo
from tsp_problem import TSPProblem 
from vrp_problem import VRPProblem
from vrp_solution import VRPSolution
import DWaveSolvers
import networkx as nx

# Attributes : VRPProblem
class FullQuboSolver:
    def __init__(self, problem):
        self.problem = problem

    def solve(self, only_one_const, order_const, capacity_const, solver_type = 'qbsolv', num_reads = 50):
        dests = len(self.problem.dests)
        vehicles = len(self.problem.capacities)
        limits = [dests for _ in range(vehicles)]
        vrp_qubo = self.problem.get_qubo_with_limits(limits, only_one_const, order_const, capacity_const)
        samples = DWaveSolvers.solve_qubo(vrp_qubo, solver_type = solver_type, num_reads = num_reads)
        sample = samples[0]
        solution = VRPSolution(self.problem, sample, limits)
        return solution
