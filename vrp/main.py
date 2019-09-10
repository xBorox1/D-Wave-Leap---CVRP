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
    for t in ['quantum1-1', 'quantum1-2', 'clustered1-1', 'clustered1-2', 'group1-1', 'group1-2', 'group2-1', 'group2-2', 'group3-1', 'group3-2', 'group4-1', 'group4-2', 'group5-1', 'group5-2', 'group6-1', 'group6-2', 'medium1-1', 'medium1-2', 'quantum1_75-1', 'quantum1_75-2', 'quantum1_100-1', 'quantum1_100-2', 'quantum1_150-1', 'quantum1_150-2', 'quantum1_200-1', 'quantum1_200-2']:
        print("Test : ", t)
        TEST = '../tests_cvrptw/exact/' + t + '.test'
        #OUT = '../tests_cvrptw/exact/' + t + '.test'
        test = read_test(TEST)

        # Problem parameters
        sources = test['sources']
        costs = test['costs']
        time_costs = test['time_costs']
        capacities = test['capacities']
        dests = test['dests']
        weigths = test['weights']
        time_windows = test['time_windows']

        only_one_const = 10000000.
        order_const = 1.
        capacity_const = 0.
        time_const = 0.

        problem = VRPTWProblem(sources, costs, time_costs, capacities, dests, weigths, time_windows)
        vrp_solver = SolutionPartitioningSolver(problem, DBScanSolver(problem, anti_noiser = True))
        solver = MergingTimeWindowsVRPTWSolver(problem, vrp_solver)


        result = solver.solve(only_one_const, order_const, capacity_const,
                solver_type = 'qbsolv', num_reads = 500)

        if result == None:
            print("Niestety coś poszło nie tak :(")
            continue

        print(result.solution)
        print(result.check())
        print(result.total_cost())
        print(result.all_time_costs())
        print(result.all_weights())
