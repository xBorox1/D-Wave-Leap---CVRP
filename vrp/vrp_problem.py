from qubo_helper import Qubo
from itertools import combinations, product

# VRP problem with multi-source
class VRPProblem:

    TIME_WINDOW_RADIUS = 60

    # TODO : time windows
    def __init__(self, sources, costs, time_costs, capacities, dests, weights, time_windows):
        # Merging all sources into one source.
        source = 0
        weights[source] = 0
        self.source = source
        in_nearest_sources = dict()
        out_nearest_sources = dict()

        # Finding nearest source for all destinations
        for dest in dests:
            in_nearest = sources[0]
            out_nearest = sources[0]
            for s in sources:
                costs[source][s] = 0
                costs[s][source] = 0
                if costs[s][dest] < costs[in_nearest][dest]:
                    in_nearest = s
                if costs[dest][s] < costs[dest][out_nearest]:
                    out_nearest = s
            costs[source][dest] = costs[in_nearest][dest]
            costs[dest][source] = costs[dest][out_nearest]
            time_costs[source][dest] = costs[in_nearest][dest]
            time_costs[dest][source] = costs[dest][out_nearest]
            in_nearest_sources[dest] = in_nearest
            out_nearest_sources[dest] = out_nearest
        time_costs[source][source] = 0

        self.costs = costs
        self.time_costs = time_costs
        self.capacities = capacities
        self.dests = dests
        self.weights = weights
        self.in_nearest_sources = in_nearest_sources
        self.out_nearest_sources = out_nearest_sources
        self.time_windows = time_windows

    def get_capacity_qubo(self, capacity, start_step, final_step):
        dests = self.dests
        weights = self.weights
        cap_qubo = Qubo()

        for (d1, d2) in combinations(dests, 2):
            for (s1, s2) in combinations(range(start_step, final_step + 1), 2):
                index = ((s1, d1), (s2, d2))
                cost = weights[d1] * weights[d2] / capacity**2
                cap_qubo.add(index, cost)

        return cap_qubo

    def get_time_qubo(self, start_step, final_step):
        dests = self.dests
        time_windows = self.time_windows
        tim_qubo = Qubo()

        for (d1, d2) in product(dests, dests):
            for (s1, s2) in combinations(range(start_step, final_step + 1), 2):
                index = ((s1, d1), (s2, d2))

                time_diff = time_windows[d1] - time_windows[d2]
                step_diff = s2 - s1
                cost = float(time_diff) / float(step_diff)
                if time_diff < 0:
                    cost = 0

                tim_qubo.add(index, cost)

        return tim_qubo

    def get_order_qubo(self, start_step, final_step, dests):
        costs = self.costs
        source = self.source
        ord_qubo = Qubo()

        # Order constraint
        for step in range(start_step, final_step):
            for dest1 in dests:
                for dest2 in dests:
                    cost = costs[dest1][dest2]
                    index = ((step, dest1), (step + 1, dest2))
                    ord_qubo.add(index, cost)

        # First and last vertices
        for dest in dests:
            in_index = ((start_step, dest), (start_step, dest))
            out_index = ((final_step, dest), (final_step, dest))
            in_cost = costs[source][dest]
            out_cost = costs[dest][source]
            ord_qubo.add(in_index, in_cost)
            ord_qubo.add(out_index, out_cost)

        return ord_qubo

    # All vehicles have number od destinations.
    def get_qubo_with_partition(self, vehicle_partitions,
            only_one_const, order_const, capacity_const, time_const):
        limits = [(r, r) for r in vehicle_partitions]
        return self.get_qubo_with_both_limits(limits,
                only_one_const, order_const, capacity_const, time_const)

    def get_qubo_with_limits(self, vehicle_limits,
            only_one_const, order_const, capacity_const, time_const):
        limits = [(0, r) for r in vehicle_limits]
        return self.get_qubo_with_both_limits(limits,
                only_one_const, order_const, capacity_const, time_const)

    # vehicles_limits - list of pairs (a, b), a <= b
    # All vehicles have limit of orders.
    # To do that we have more steps and vehicles can 'wait' in source.
    def get_qubo_with_both_limits(self, vehicle_limits,
            only_one_const, order_const, capacity_const, time_const):
        steps = 0
        for (_, r) in vehicle_limits:
            steps += r

        capacities = self.capacities
        dests = self.dests
        source = self.source
        dests_with_source = dests.copy()
        dests_with_source.append(source)
        costs = self.costs
        vrp_qubo = Qubo()

        # Only one step for one destination.
        for dest in self.dests:
            vrp_qubo.add_only_one_constraint([(step, dest) for step in range(steps)], only_one_const) 

        start = 0
        for vehicle in range(len(vehicle_limits)):
            min_size = vehicle_limits[vehicle][0]
            max_size = vehicle_limits[vehicle][1]
            min_final = start + min_size - 1
            max_final = start + max_size - 1

            # First steps should have normal destinations.
            if min_size != 0:
                for step in range(start, min_final + 1):
                    vrp_qubo.add_only_one_constraint([(step, dest) for dest in dests], only_one_const) 
                ord_min_qubo = self.get_order_qubo(start, min_final, dests)
                vrp_qubo.merge_with(ord_min_qubo, 1., order_const)

            # In other steps vehicles can wait in source.
            if max_size != min_size:
                for step in range(min_final + 1, max_final + 1):
                    vrp_qubo.add_only_one_constraint([(step, dest) for dest in dests_with_source], only_one_const)
                ord_max_qubo = self.get_order_qubo(min_final + 1, max_final, dests_with_source)
                vrp_qubo.merge_with(ord_max_qubo, 1., order_const)

            # From min_final step to min_final + 1 step
            if min_size != 0 and min_size != max_size:
                for dest1 in dests:
                    for dest2 in dests_with_source:
                        cost = costs[dest1][dest2]
                        index = ((min_final, dest1), (min_final + 1, dest2))
                        vrp_qubo.add(index, cost * order_const)

            # Capacity constraints
            if capacity_const != 0:
                capacity = capacities[vehicle]
                cap_qubo = self.get_capacity_qubo(capacity, start, max_final)
                vrp_qubo.merge_with(cap_qubo, 1., capacity_const)

            if time_const != 0:
                tim_qubo = self.get_time_qubo(start, max_final)
                vrp_qubo.merge_with(tim_qubo, 1., time_const)

            start = max_final + 1

        return vrp_qubo       
