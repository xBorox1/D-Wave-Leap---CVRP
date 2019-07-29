from qubo_helper import Qubo
from itertools import combinations, product

# VRP problem with multi-source
# Attributes : sources list, costs dict/matrix, capacities matrix, destinations list, weights of orders
class VRPProblem:

    # TODO : time windows
    def __init__(self, sources, costs, capacities, dests, weights):
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
                if costs[s][dest] < costs[in_nearest][dest]:
                    in_nearest = s
                if costs[dest][s] < costs[dest][out_nearest]:
                    out_nearest = s
            costs[source][dest] = costs[in_nearest][dest]
            costs[dest][source] = costs[dest][out_nearest]
            in_nearest_sources[dest] = in_nearest
            out_nearest_sources[dest] = out_nearest
        costs[source][source] = 0

        self.costs = costs
        self.capacities = capacities
        self.dests = dests
        self.weights = weights
        self.in_nearest_sources = in_nearest_sources
        self.out_nearest_sources = out_nearest_sources

    def get_order_qubo(self, start_step, final_step):
        dests = self.dests
        costs = self.costs
        ord_qubo = Qubo()

        for step in range(start_step, final_step):
            for dest1 in dests:
                for dest2 in dests:
                    cost = costs[dest1][dest2]
                    index = ((step, dest1), (step + 1, dest2))
                    ord_qubo.add(index, cost * order_const)

        return ord_qubo

    def get_capacity_qubo(self, capacity, start_step, final_step):
        dests = self.dests
        weights = self.weights
        cap_qubo = Qubo()

        for (d1, d2) in combinations(dests, 2):
            for (s1, s2) in combinations(range(start_step, final_step), 2):
                index = ((s1, d1), (s2, d2))
                cost = weights[d1] * weights[d2] / capacity**2
                cap_qubo.add(index, cost)

        return cap_qubo

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
    def get_qubo_with_partition(self, vehicles_partition, only_one_const, order_const, capacity_const):
        costs = self.costs
        capacities = self.capacities
        dests = self.dests    
        steps = len(dests)

        vrp_qubo = Qubo()

        # Only one vertex for one step.
        for step in range(steps):
            vrp_qubo.add_only_one_constraint([(step, dest) for dest in dests], only_one_const)

        # Only one step for one vertex
        for dest in self.dests:
            vrp_qubo.add_only_one_constraint([(step, dest) for step in range(steps)], only_one_const) 

        start = 0
        for vehicle in range(len(vehicles_partition)):
            size = vehicles_partition[vehicle]
            final = start + size - 1

            ord_qubo = self.get_order_qubo(start, final, dests)
            vrp_qubo.merge_with(ord_qubo, 1., order_const)

            # Capacity constraints
            if capacity_const != 0:
                capacity = capacities[vehicle]
                cap_qubo = self.get_capacity_qubo(capacity, start, final)
                vrp_qubo.merge_with(cap_qubo, 1., capacity_const)

            start = final + 1

        return vrp_qubo

    # All vehicles have limit of orders.
    # To do that we have more steps and vehicles can 'wait' in source.
    def get_qubo_with_limits(self, vehicles_limits, only_one_const, order_const, capacity_const):
        steps = 0
        for limit in vehicles_limits:
            steps += limit
        dests_num = len(self.dests)

        capacities = self.capacities
        dests = self.dests
        source = self.source
        dests_with_source = dests.copy()
        dests_with_source.append(source)
        vrp_qubo = Qubo()

        # Only one destination for one step.
        for step in range(steps):
            vrp_qubo.add_only_one_constraint([(step, dest) for dest in dests_with_source], only_one_const)

        # Only one step for one destination.
        for dest in self.dests:
            vrp_qubo.add_only_one_constraint([(step, dest) for step in range(steps)], only_one_const) 

        start = 0
        for vehicle in range(len(vehicles_limits)):
            size = vehicles_limits[vehicle]
            final = start + size - 1

            ord_qubo = self.get_order_qubo(start, final, dests_with_source)
            vrp_qubo.merge_with(ord_qubo, 1., order_const)

            # Capacity constraints
            if capacity_const != 0:
                capacity = capacities[vehicle]
                cap_qubo = self.get_capacity_qubo(capacity, start, final)
                vrp_qubo.merge_with(cap_qubo, 1., capacity_const)

            start = final + 1

        return vrp_qubo

    # Returns list of lists of destinations.
    # Doesn't include magazines.
    def decode_answer_with_partition(self, sample, vehicles_partition):
        result = list()
        vehicle_result = list()
        step = 0
        vehicle = 0

        for (s, dest) in sample:
            if sample[(s, dest)] == 1:
                vehicle_result.append(dest)
                step += 1
                if vehicles_partition[vehicle] == step:
                    result.append(vehicle_result)
                    step = 0
                    vehicle += 1
                    vehicle_result = list()
                    if len(vehicles_partition) <= vehicle:
                        return result

        return result

    def decode_answer_with_limits(self, sample, vehicles_limits):
        result = list()
        vehicle_result = list()
        step = 0
        vehicle = 0

        for (s, dest) in sample:
            if sample[(s, dest)] == 1:
                if dest != 0:
                    vehicle_result.append(dest)
                step += 1
                if vehicles_limits[vehicle] == step:
                    result.append(vehicle_result)
                    step = 0
                    vehicle += 1
                    vehicle_result = list()
                    if len(vehicles_limits) <= vehicle:
                        return result

        return result

    # Checks capacity and visiting.
    def check_answer(self, answer):
        capacities = self.capacities
        weights = self.weights
        vehicle_num = 0

        for vehicle_dests in answer:
            cap = self.capacities[vehicle_num]
            for dest in vehicle_dests:
                cap -= weights[dest]
            if cap < 0: 
                return False

        dests = self.dests
        answer_dests = [dest for vehicle_dests in answer for dest in vehicle_dests]
        if len(dests) != len(answer_dests):
            return False

        lists_cmp = set(dests) & set(answer_dests)
        if lists_cmp == len(dests):
            return False

        return True

    def answer_cost(self, answer):
        costs = self.costs
        source = self.source
        cost = 0

        for vehicle_dests in answer:
            prev = source
            for dest in vehicle_dests:
                cost += costs[prev][dest]
                prev = dest
            cost += costs[prev][source]

        return cost

    # Returns answer with magazines.
    def get_full_answer(self, answer):
        result = list()
        for vehicle_dests in answer:
            l = vehicle_dests.copy()
            if len(l) != 0:
                l.insert(0, self.in_nearest_sources[l[0]])
                l.append(self.out_nearest_sources[l[len(l) - 1]])
            result.append(l)
        return result

