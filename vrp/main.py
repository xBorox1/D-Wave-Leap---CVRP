from qubo_helper import Qubo
from tsp_problem import TSPProblem 
from vrp_problem import VRPProblem
import DWaveSolvers
import networkx as nx

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
    partition = [2, 3]
    only_one_const = 100.
    order_const = 1.
    capacity_const = 1.

    # Creating qubo
    problem = VRPProblem(sources, paths, capacities, dests, weights)
    qub = problem.get_qubo_with_partition(partition, only_one_const, order_const, capacity_const)

    samples = DWaveSolvers.solve_qubo_on_cpu(qub, limit = 1)
    # samples = DWaveSolvers.solve_qubo_on_qpu(qub)
    first = samples[0]
    answer = problem.decode_answer_with_partition(first, partition)
    if problem.check_answer(answer):
        print("Cost : ", problem.answer_cost(answer))
        print(problem.get_full_answer(answer))
