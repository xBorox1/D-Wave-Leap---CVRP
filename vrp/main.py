from qubo_helper import Qubo
from tsp_problem import TSPProblem 
from vrp_problem import VRPProblem
from vrptw_problem import VRPTWProblem
from vrptw_solvers import *
from vrp_solvers import *
from itertools import product
import DWaveSolvers
import networkx as nx
import numpy as np

if __name__ == '__main__':

    # Some graph
    n = 20
    """paths = np.random.random_integers(1, 10, (n, n))

    for u in range(n):
        paths[u][u] = 0

    for u in range(n):
        for (i, j) in product(range(n), range(n)):
            paths[i][j] = min(paths[i][j], paths[i][u] + paths[u][j])"""

    paths = np.zeros((n, n))
    for (i, j) in product(range(n), range(n)):
        paths[i][j] = abs(i - j)

    # Problem parameters
    sources = [10]
    #sources = [0, 3, 15, 50, 77, 38, 89]
    costs = paths
    time_costs = costs
    #capacities = [n, n, n, n, n, n, n, n, n, n]
    capacities = [n, n]
    #dests = [1, 2, 16, 19, 8, 25, 55, 33, 31, 88, 97, 24, 10, 61, 48, 11, 92, 54, 38, 65]
    dests = [1, 2, 4, 8, 12, 14, 16, 19]
    weights = [1 for _ in range(0, n)]

    time_windows = dict()
    time_windows[1] = 10
    time_windows[2] = 10
    time_windows[4] = 15
    time_windows[8] = 15
    time_windows[12] = 10
    time_windows[14] = 10
    time_windows[16] = 15
    time_windows[19] = 15

    only_one_const = 100.
    order_const = 1.
    capacity_const = 0.
    time_const = 0.

    problem = VRPTWProblem(sources, costs, time_costs, capacities, dests, weights, time_windows)
    solver = MergingTimeWindowsVRPTWSolver(problem)
    vrp_solver = AveragePartitionSolver(None)

    print(solver.solve(only_one_const, order_const, capacity_const,
            vrp_solver, solver_type = 'standard', num_reads = 50))

    """solver = FullQuboSolver(problem)
    #solver = AveragePartitionSolver(problem)

    result = solver.solve(only_one_const, order_const, capacity_const, time_const,
            solver_type = 'qbsolv', num_reads = 100)
    print(result.solution)
    #result.description()"""
