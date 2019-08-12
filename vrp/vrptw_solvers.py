from qubo_helper import Qubo
from tsp_problem import TSPProblem 
from vrp_problem import VRPProblem
from vrp_solution import VRPSolution
from itertools import product
from scipy.optimize import linear_sum_assignment
import DWaveSolvers
import numpy as np

# Attributes : VRPProblem
class VRPTWSolver:

    TIME_WINDOW_RADIUS = 10
    TIME_BLOCK = 10
    TIME_WINDOWS_DIFF = 5

    def __init__(self, problem):
        self.problem = problem

    def solve(self, only_one_const, order_const, capacity_const, time_const,
            solver_type = 'qbsolv', num_reads = 50):
        pass

class MergingTimeWindowsVRPTWSolver(VRPTWSolver):

    INF = 1000000000

    def __init__(self, problem):
        self.problem = problem

    def _merge_one(self, first_sample, second_sample, time):
        if len(first_sample) == 0:
            return second_sample

        costs = self.problem.costs
        time_costs = self.problem.time_costs
        fir = np.asarray(first_sample)
        sec = np.asarray([0] + second_sample)

        n, m = len(fir), len(sec)

        # Time to exact state.
        # TODO : optimize first destination for dp
        time_lapse = 0.
        first_order = 0
        while time_lapse < time and first_order != n - 1:
            time_lapse += time_costs[fir[first_order]][fir[first_order + 1]]
            first_order += 1

        first_order += 1

        result = fir[:first_order].tolist()

        fir = np.append([0], fir[first_order:])
        n = len(fir)

        # Dynamic programming.
        best_solution = np.zeros((n + 1, m + 1, 2), dtype=float)
        prev_solution = np.zeros((n + 1, m + 1, 2), dtype=int)

        best_solution[1][0][1] = self.INF
        best_solution[0][1][0] = self.INF

        for i in range(2, n):
            best_solution[i][0][0] = best_solution[i-1][0][0] + costs[fir[i-1]][fir[i]]
            best_solution[i][0][1] = self.INF
            prev_solution[i][0][0] = 0 

        for j in range(2, m):
            best_solution[0][j][1] = best_solution[0][j-1][1] + costs[sec[j-1]][sec[j]]
            best_solution[0][j][0] = self.INF
            prev_solution[0][j][1] = 1

        for i in range(1, n):
            for j in range(1, m):
                sol1 = best_solution[i-1][j][0] + costs[fir[i-1]][fir[i]]
                sol2 = best_solution[i-1][j][1] + costs[sec[j]][fir[i]]
                sol3 = best_solution[i][j-1][0] + costs[fir[i]][sec[j]]
                sol4 = best_solution[i][j-1][1] + costs[sec[j-1]][sec[j]]

                if i == 1:
                    sol1 = self.INF
                if j == 1:
                    sol4 = self.INF

                if sol1 > sol2:
                    best_solution[i][j][0] = sol2
                    prev_solution[i][j][0] = 1
                else:
                    best_solution[i][j][0] = sol1
                    prev_solution[i][j][0] = 0

                if sol3 > sol4:
                    best_solution[i][j][1] = sol4
                    prev_solution[i][j][1] = 1
                else:
                    best_solution[i][j][1] = sol3
                    prev_solution[i][j][1] = 0

        # TODO : change this
        if sec[m - 1] == 0:
            best_solution[n - 1][m - 1][0] = self.INF

        dp_result = list()
        state = (n - 1, m - 1, int(0))
        if best_solution[n - 1][m - 1][0] > best_solution[n - 1][m - 1][1]:
            state = (n - 1, m - 1, 1)

        while state != (0, 0, 0) and state != (0, 0, 1):
            s1 = state[0]
            s2 = state[1]
            s3 = state[2]
            if s3 == 0:
                dp_result.append(fir[s1])
                state = (s1 - 1, s2, prev_solution[s1][s2][s3])
            else:
                dp_result.append(sec[s2])
                state = (s1, s2 - 1, prev_solution[s1][s2][s3])

        dp_result = list(reversed(dp_result))
        result = result + dp_result

        return result


    def _merge_all(self, first_solution, second_solution, time):
        if first_solution == None:
            return second_solution

        """result = list()
        for i in range(len(first_solution)):
            result.append(self._merge_one(first_solution[i], second_solution[i], time))

        return result"""

        # Finding best matching.
        size = len(first_solution)
        matching_costs = np.zeros((size, size), dtype=int)
        for (i, j) in product(range(size), range(size)):
            merging = self._merge_one(first_solution[i], second_solution[j], time)
            weight = VRPSolution(self.problem, None, None, 
                    solution = [merging]).total_cost()
            matching_costs[i][j] = weight

        row_matching, col_matching = linear_sum_assignment(matching_costs)
        mates = np.zeros((size), dtype=int)
        for i in range(size):
            mates[col_matching[i]] = i

        result = list()
        for i in range(size):
            mate = row_matching[mates[i]]
            merging = self._merge_one(first_solution[i], second_solution[mate], time)
            result.append(merging)

        return result

    def solve(self, only_one_const, order_const, capacity_const,
            vrp_solver, solver_type = 'qbsolv', num_reads = 50):
        problem = self.problem
        time_windows = problem.time_windows
        dests = problem.dests
        dests_blocks = dict()

        min_time = self.INF
        max_time = 0

        for dest in dests:
            time = time_windows[dest]
            min_time = min(time, min_time)
            max_time = max(time, max_time)

        for time in range(min_time, max_time + 1, self.TIME_WINDOWS_DIFF):
            dests_blocks[time] = list()

        for dest in dests:
            dests_blocks[time].append(dest)

        solution = None
        for time in range(min_time, max_time + 1, self.TIME_WINDOWS_DIFF):
            dests = dests_blocks[time]

            first_source = (time == min_time)
            last_source = (time == max_time)

            vrp_problem = VRPProblem([problem.source], problem.costs, problem.time_costs,
                    problem.capacities, dests, problem.weights, first_source = first_source, 
                    last_source = last_source)
            vrp_solver.set_problem(vrp_problem)
            next_solution = vrp_solver.solve(only_one_const, order_const, capacity_const,
                        solver_type = solver_type, num_reads = num_reads).solution
            solution = self._merge_all(solution, next_solution, time - self.TIME_WINDOW_RADIUS)

        return solution
