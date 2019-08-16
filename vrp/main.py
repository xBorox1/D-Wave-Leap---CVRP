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
from input import *

CAPACITY = 1000

if __name__ == '__main__':

    for i in range(1):
        TEST = '../tests_vrp/exact/medium/medium-' + str(i) + '.test'
        test = read_test(TEST)

        # Problem parameters
        sources = test['sources']
        costs = test['costs']
        time_costs = test['time_costs']
        capacities = [CAPACITY, CAPACITY]
        dests = test['dests']
        weigths = test['weights']
        time_windows = test['time_windows']

        only_one_const = 1000000.
        order_const = 1.
        capacity_const = 0.
        time_const = 0.

        #problem = VRPTWProblem(sources, costs, time_costs, capacities, dests, weigths, time_windows)
        #solver = MergingTimeWindowsVRPTWSolver(problem)
        #vrp_solver = AveragePartitionSolver(None)

        #solution = solver.solve(only_one_const, order_const, capacity_const,
        #            vrp_solver, solver_type = 'qbsolv', num_reads = 1000)
        #print(solution)

        problem = VRPProblem(sources, costs, time_costs, capacities, dests, weigths)
        #solver = FullQuboSolver(problem)
        #solver = AveragePartitionSolver(problem)
        solver = DBScanSolver(problem)

        result = solver.solve(only_one_const, order_const, capacity_const,
                solver_type = 'qbsolv')
        print(result.solution)
        print(result.check())
        print(result.total_cost())
