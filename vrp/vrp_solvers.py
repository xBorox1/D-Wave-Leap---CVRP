from qubo_helper import Qubo
from tsp_problem import TSPProblem 
from vrp_problem import VRPProblem
import DWaveSolvers
import networkx as nx

# Attributes : VRPProblem
class FullQuboSolver:
    def __init__(self, problem):
        self.problem = problem

    def solve(self, only_one_const, order_const, capacity_const, solve_type = 'cpu'):
        dests = len(self.problem.dests)
        vehicles = len(self.problem.capacities)
        limits = [dests for _ in range(vehicles)]
        vrp_qubo = self.problem.get_qubo_with_limits(limits, only_one_const, order_const, capacity_const)
        if solve_type == 'cpu':
            samplers = DWaveSolvers.solve_qubo_on_cpu(vrp_qubo)
        else:
            samplers = DWaveSolvers.solve_qubo_on_qpu(vrp_qubo)
        sample = samplers[0]
        answer = self.problem.get_full_answer(self.problem.decode_answer_with_limits(sample, limits))
        return answer
