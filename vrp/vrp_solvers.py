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
    MAX_WEIGHT = 1000

    def __init__(self, problem, anti_noiser = True):
        self.problem = problem
        self.anti_noiser = anti_noiser

    def _range_query(self, dests, costs, source, radius):
        result = list()
        for dest in dests:
            if (costs[source][dest] + costs[dest][source]) / 2 <= radius:
                result.append(dest)
        return result

    def _dbscan(self, dests, costs, radius, min_size):
        clusters_num = -1

        states = dict()
        # Undifined cluster.
        for d in dests:
            states[d] = -2

        for d in dests:
            neighbours = self._range_query(dests, costs, d, radius)
            if len(neighbours) < min_size:
                states[d] = -1

        for dest in dests:
            if states[dest] != -2:
                continue

            clusters_num += 1
            q = Queue()
            q.put(dest)

            while not q.empty():
                dest2 = q.get()
                states[dest2] = clusters_num
                neighbours = self._range_query(dests, costs, dest2, radius)
                for v in neighbours:
                    if states[v] == -2:
                        q.put(v)

        for dest in dests: 
            if states[dest] == -1:
                min_dist = self.MAX_DIST
                best_neighbour = -1
                for d in dests:
                    if states[d] != -1:
                        if costs[d][dest] < min_dist:
                            best_neighbour = d
                            min_dist = costs[d][dest]
                if best_neighbour == -1:
                    clusters_num += 1
                    states[dest] = clusters_num
                else:
                    states[dest] = states[best_neighbour]

        clusters = list()
        for i in range(clusters_num + 1):
            clusters.append(list())
        for dest in dests:
            cl = states[dest]
            clusters[cl].append(dest)

        return clusters

    def _recursive_dbscan(self, dests, costs, min_radius, max_radius,
                          clusters_num, max_len, max_weight):
        best_res = [[d] for d in dests]

        min_r = min_radius
        max_r = max_radius
        curr_r = max_r

        while min_r + 1 < max_r:
            curr_r = (min_r + max_r) / 2

            # TODO : min_size parameter
            clusters = self._dbscan(dests, costs, curr_r, 1)

            if len(clusters) < clusters_num:
                max_r = curr_r
            else:
                min_r = curr_r
                if len(clusters) < len(best_res):
                    best_res = clusters

        for cluster in best_res:
            weight = 0
            for dest in cluster:
                weight += self.problem.weigths[dest]
            if len(cluster) > max_len or weight > max_weight:
                best_res.remove(cluster)
                # TODO : clusters_num parameter : 2 or clusters_num ?
                best_res += self._recursive_dbscan(cluster, costs, 0., self.MAX_DIST, 2,
                                                   max_len, max_weight)

        if self.anti_noiser:
            while len(best_res) > clusters_num:
                singleton = [0]
                for cluster in best_res:
                    if len(cluster) == 1:
                        singleton = cluster
                        break

                if singleton == [0]:
                    break

                best_res.remove(singleton)

                one = singleton[0]
                best_cluster = []
                best_dist = self.MAX_DIST

                for cluster in best_res:
                    if len(cluster) == max_len or cluster == singleton:
                        continue

                    weight = 0
                    min_dist = self.MAX_DIST

                    for dest in cluster:
                        weight += self.problem.weigths[dest]
                        min_dist = min(min_dist, costs[dest][one])
                    if weight + self.problem.weigths[one] <= max_weight:
                        if best_dist > min_dist:
                            best_dist = min_dist
                            best_cluster = cluster

                if best_cluster == []:
                    best_res.append(singleton)
                    break
                best_res.remove(best_cluster)
                best_res.append(best_cluster + singleton)

        return best_res

    # My clusterring algorithm which isn't effective.
    """def _find_set(self, dests, costs, dest, radius):
        result = list()
        q = Queue()
        q.put(dest)

        while not q.empty():
            d = q.get()
            if d in result:
                continue
            result.append(d)
            neighbours = self._range_query(dests, costs, d, radius)
            for n in neighbours:
                q.put(n)

        return result

    def _best_neighbours(self, dests, costs, dest, cluster_size):
        min_r = 0.
        max_r = self.MAX_DIST
        best_res = [dest]

        while min_r + 1 < max_r:
            curr_r = (min_r + max_r) / 2
            cluster = self._find_set(dests, costs, dest, curr_r)

            if len(cluster) <= cluster_size:
                min_r = curr_r
                if len(cluster) > len(best_res):
                    best_res = cluster
            else:
                max_r = curr_r

        return best_res

    def _divide_set(self, dests, source, costs, clusters_num):
        if clusters_num == 1:
            return [dests]

        clusters_size = int((len(dests) + clusters_num - 1) / clusters_num)

        best_cost = -self.MAX_DIST
        best_dest = 0

        for (d1, d2) in product(dests, dests):
            if d1 == d2:
                continue
            new_cost = costs[source][d1] - costs[d1][d2] - costs[d2][source]
            if  new_cost > best_cost:
                best_cost = new_cost
                best_dest = d1

        best_neighbours = self._best_neighbours(dests, costs, best_dest, clusters_size)
        new_dests = dests.copy()
        for d in best_neighbours:
            new_dests.remove(d)

        return [best_neighbours] + self._divide_set(new_dests, source, costs, clusters_num - 1)

    def _recursive_divide_set(self, dests, source, costs, clusters_num, max_len):
        clusters = self._divide_set(dests, source, costs, clusters_num)
        for cluster in clusters:
            if len(cluster) > max_len:
                clusters.remove(cluster)
                clusters += self._recursive_divide_set(cluster, source, costs,
                                        int((len(cluster) + max_len - 1) / max_len), max_len)
        return clusters"""

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

        clusters = self._recursive_dbscan(dests, costs, 0., self.MAX_DIST, vehicles, self.MAX_LEN, capacities[0])

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
            new_costs[i][j] = costs[id1][id2]
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

class SolutionPartitioningSolver(VRPSolver):

    INF = 1000000000

    def __init__(self, problem, solver):
        self.problem = problem
        self.solver = solver
    
    def _divide_solution_greedy(self, solution):
        solution = solution[1:-1]
        problem = self.problem
        capacities = problem.capacities
        costs = problem.costs
        weights = problem.weigths

        new_solution = []
        pointer = 0
        dests = len(solution)

        for cap in reversed(capacities):
            actual_cap = cap
            sub_dests = []
            if pointer != dests:
                sub_dests = [0]
            while pointer < dests and actual_cap >= weights[solution[pointer]]:
                actual_cap -= weights[solution[pointer]]
                sub_dests.append(solution[pointer])
                pointer += 1
            if len(sub_dests) > 0:
                sub_dests.append(0)
            new_solution.append(sub_dests)

        new_solution.reverse()
        return VRPSolution(problem, None, None, new_solution)

    def _divide_solution_greedy_dp(self, solution):
        problem = self.problem
        capacities = problem.capacities
        costs = problem.costs
        weights = problem.weigths

        dests = len(solution)
        vehicles = len(capacities)
        div_costs = np.zeros(dests)
        for i in range(1, dests - 1):
            d1 = solution[i]
            d2 = solution[i+1]
            div_costs[i] = costs[d1][0] + costs[0][d2] - costs[d1][d2]

        dp = np.zeros((dests, vehicles + 1), dtype=float)
        prev_state = np.zeros((dests, vehicles + 1), dtype=int)

        for i in range(dests):
            if i != 0:
                dp[i][0] = self.INF
            for j in range(1, vehicles + 1):
                cap = capacities[j-1]
                pointer = i
                dp[i][j] = dp[i][j-1]
                prev_state[i][j] = i
                while pointer > 0 and cap > weights[solution[pointer]]:
                    pointer -= 1
                    new_cost = div_costs[pointer] + dp[pointer][j-1]
                    if new_cost < dp[i][j]:
                        dp[i][j] = new_cost
                        prev_state[i][j] = pointer
                    cap -= weights[solution[pointer + 1]]

        new_solution = []
        pointer = dests - 1
        for j in reversed(range(1, vehicles + 1)):
            prev = prev_state[pointer][j]
            if prev != pointer:
                lis = solution[(prev + 1):(pointer + 1)]
                if prev != -1:
                    lis = [0] + lis
                if pointer != dests - 1:
                    lis = lis + [0]
                new_solution.append(lis)
            else:
                new_solution.append([])
            pointer = prev
        
        new_solution.reverse()
        return VRPSolution(problem, None, None, new_solution)

    def solve(self, only_one_const, order_const, capacity_const,
            solver_type = 'qbsolv', num_reads = 50):
        problem = self.problem
        capacity = 0
        weights = problem.weigths
        for w in weights:
            capacity += w

        # Creating new problem with one vehicle
        sources = [0]
        dests = problem.dests
        costs = problem.costs
        time_costs = problem.time_costs
        new_capacities = [capacity]
        new_problem = VRPProblem(sources, costs, time_costs, new_capacities, dests, weights)

        solver = self.solver
        solver.set_problem(new_problem)
        solution = solver.solve(only_one_const, order_const, capacity_const,
            solver_type = 'qbsolv', num_reads = 50)

        sol = solution.solution[0]
        return self._divide_solution_greedy_dp(sol)
