from qubo_helper import Qubo
from tsp_problem import TSPProblem 
from vrp_problem import VRPProblem
import DWaveSolvers
import networkx as nx
from vrp_solvers import FullQuboSolver

if __name__ == '__main__':

    # Some graph
    n = 20
    G = nx.cycle_graph(n)
    paths = dict(nx.all_pairs_shortest_path(G))
    for i in range(n):
        for j in range(n):
            paths[i][j] = len(paths[i][j]) - 1

    # Problem parameters
    sources = [0, 3, 15]
    costs = paths
    capacities = [n, n]
    dests = [1, 2, 16, 19, 8]
    weights = [1 for _ in range(0, n)]
    limits = [5, 5]
    only_one_const = 100.
    order_const = 1.
    capacity_const = 1.

    problem = VRPProblem(sources, paths, capacities, dests, weights)
    solver = FullQuboSolver(problem)
    print(solver.solve(only_one_const, order_const, capacity_const))
