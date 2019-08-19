from qubo_helper import Qubo
from tsp_problem import TSPProblem 
from vrp_problem import VRPProblem
from vrp_solution import VRPSolution
from itertools import product
import DWaveSolvers
import networkx as nx
import numpy as np
from queue import Queue

# Attributes : VRPProblem
class VRPSolver:
    def __init__(self, problem):
        self.problem = problem

    def set_problem(self, problem):
        self.problem = problem

    def solve(self, only_one_const, order_const, capacity_const,
            solver_type = 'qbsolv', num_reads = 50):
        pass

class FullQuboSolver(VRPSolver):
    def solve(self, only_one_const, order_const, capacity_const,
            solver_type = 'qbsolv', num_reads = 50):
        dests = len(self.problem.dests)
        vehicles = len(self.problem.capacities)

        limits = [dests for _ in range(vehicles)]

        vrp_qubo = self.problem.get_qubo_with_limits(limits, only_one_const, order_const, capacity_const)
        samples = DWaveSolvers.solve_qubo(vrp_qubo, solver_type = solver_type, num_reads = num_reads)
        sample = samples[0]
        solution = VRPSolution(self.problem, sample, limits)
        return solution

class AveragePartitionSolver(VRPSolver):
    def solve(self, only_one_const, order_const, capacity_const,
            solver_type = 'qbsolv', num_reads = 50, limit_radius = 1):
        dests = len(self.problem.dests)
        vehicles = len(self.problem.capacities)

        avg = int(dests / vehicles)

        limits = [(max(avg - limit_radius, 0), min(avg + limit_radius, dests)) for _ in range(vehicles)]
        max_limits = [r for (_, r) in limits]

        vrp_qubo = self.problem.get_qubo_with_both_limits(limits,
                only_one_const, order_const, capacity_const)

        samples = DWaveSolvers.solve_qubo(vrp_qubo, solver_type = solver_type, num_reads = num_reads)
        sample = samples[0]

        solution = VRPSolution(self.problem, sample, max_limits)
        return solution

class DBScanSolver(VRPSolver):

    MAX_DIST = 10000000.
    MAX_LEN = 10

    def _range_query(self, dests, costs, source, radius):
        result = list()
        for dest in dests:
            if costs[source][dest] <= radius:
                result.append(dest)
        return result

    def _dbscan(self, dests, costs, radius, min_size):
        clusters_num = -1

        states = dict()
        # Undifined cluster.
        for d in dests:
            states[d] = -2

        for dest in dests:
            if states[dest] != -2:
                continue

            neighbours = self._range_query(dests, costs, dest, radius)
            if len(neighbours) < min_size:
                states[dest] = -1
                continue

            clusters_num += 1
            states[dest] = clusters_num
            q = Queue()
            for d in neighbours:
                q.put(d)

            while not q.empty():
                dest2 = q.get()
                if states[dest2] == -1:
                    states[dest2] = clusters_num
                if states[dest2] != -2:
                    continue
                states[dest2] = clusters_num
                neighbours = self._range_query(dests, costs, dest2, radius)
                if len(neighbours) >= min_size:
                    for v in neighbours:
                        q.put(v)

        clusters = list()
        for i in range(clusters_num + 1):
            clusters.append(list())
        for dest in dests:
            cl = states[dest]
            clusters[cl].append(dest)

        return clusters

    def _recursive_dbscan(self, dests, costs, min_radius, max_radius, clusters_num, max_len):
        best_res = [[d for d in dests]]

        while len(best_res) != clusters_num and min_radius + 1 < max_radius:
            curr_radius = (min_radius + max_radius) / 2

            clusters = self._dbscan(dests, costs, curr_radius, 0)

            if len(clusters) < clusters_num:
                max_radius = curr_radius
            else:
                min_radius = curr_radius
                best_res = clusters

        for cluster in best_res:
            if len(cluster) > max_len:
                best_res.remove(cluster)
                best_res += self._recursive_dbscan(cluster, costs, 0., self.MAX_DIST, max(clusters_num, 2), max_len)

        return best_res

    def solve(self, only_one_const, order_const, capacity_const,
            solver_type = 'qbsolv', num_reads = 50):
        problem = self.problem
        dests = problem.dests
        costs = problem.costs
        time_costs = problem.time_costs
        sources = [problem.source]
        capacities = problem.capacities
        weigths = problem.weigths
        vehicles = len(problem.capacities)

        # Some idea
        #if len(dests) <= self.MAX_LEN:
        #    solver = AveragePartitionSolver(problem)
        #    result = solver.solve(only_one_const, order_const, capacity_const,
        #                        solver_type = solver_type, num_reads = num_reads).solution
        #    return VRPSolution(problem, None, None, result)

        clusters = self._recursive_dbscan(dests, costs, 0, self.MAX_DIST, vehicles, self.MAX_LEN)

        if len(clusters) == vehicles:
            result = list()
            for cluster in clusters:
                new_problem = VRPProblem(sources, costs, time_costs, [capacities[0]], cluster, weigths)
                solver = FullQuboSolver(new_problem)
                solution = solver.solve(only_one_const, order_const, capacity_const,
                                    solver_type = solver_type, num_reads = num_reads).solution[0]
                result.append(solution)
            return VRPSolution(problem, None, None, result)

        solutions = list()
        solutions.append(VRPSolution(problem, None, None, [[0]]))

        for cluster in clusters:
            new_problem = VRPProblem(sources, costs, time_costs, [capacities[0]], cluster, weigths,
                                 first_source = False, last_source = False)
            solver = FullQuboSolver(new_problem)
            solution = solver.solve(only_one_const, order_const, capacity_const,
                                    solver_type = solver_type, num_reads = num_reads)
            solutions.append(solution)

        clusters_num = len(clusters) + 1
        new_dests = [i for i in range(1, clusters_num)]
        new_costs = np.zeros((clusters_num, clusters_num), dtype=float)
        new_time_costs = np.zeros((clusters_num, clusters_num), dtype=float)
        new_weigths = np.zeros((clusters_num), dtype=int)

        for (i, j) in product(range(clusters_num), range(clusters_num)):
            if i == j:
                new_costs[i][j] = 0
                time_costs[i][j] = 0
                continue
            id1 = solutions[i].solution[0][-1]
            id2 = solutions[j].solution[0][0]
            #new_costs[i][j] = solutions[j].total_cost() + costs[id1][id2]
            new_costs[i][j] = costs[id1][id2] + (solutions[i].total_cost() + solutions[j].total_cost()) / 2.
            new_time_costs[i][j] = solutions[j].all_time_costs()[0] + time_costs[id1][id2]

        for i in range(clusters_num):
            for dest in solutions[i].solution[0]:
                new_weigths[i] += weigths[dest]

        new_problem = VRPProblem(sources, new_costs, new_time_costs, capacities, new_dests, new_weigths)
        solver = DBScanSolver(new_problem)
        compressed_solution = solver.solve(only_one_const, order_const, capacity_const, 
                            solver_type = solver_type, num_reads = num_reads).solution

        uncompressed_solution = list()
        for vehicle_dests in compressed_solution:
            uncompressed = list()
            for dest in vehicle_dests:
                uncompressed += solutions[dest].solution[0]
            uncompressed_solution.append(uncompressed)

        return VRPSolution(problem, None, None, uncompressed_solution)
