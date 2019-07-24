from qubo_helper import Qubo
from itertools import combinations, product

# VRP problem with multi-source
# Attributes : sources list, costs dict/matrix, capacities matrix, destinations list, weights of orders
class VRPProblem:

    def __init__(self, sources, costs, capacities, dests, weights):
        # Merging all sources into one source.
        source = 0 # TODO : source id
        self.source = source
        in_nearest_sources = dict() # TODO : types of arguments
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

        self.costs = costs
        self.capacities = capacities
        self.dests = dests
        self.weights = weights
        self.in_nearest_sources = in_nearest_sources
        self.out_nearest_sources = out_nearest_sources

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

    # All vehicles have number od destinations.
    def get_qubo_with_partition(self, vehicles_partition, only_one_const, order_const, capacity_const):
        source = self.source
        costs = self.costs
        capacities = self.capacities
        dests = self.dests    
        weights = self.weights
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

            # Order constraint
            for step in range(start, final):
                for dest1 in dests:
                    for dest2 in dests:
                        cost = costs[dest1][dest2]
                        index = ((step, dest1), (step + 1, dest2))
                        vrp_qubo.add(index, cost * order_const)

            # First and last vertices
            for dest in dests:
                in_index = ((start, dest), (start, dest))
                out_index = ((final, dest), (final, dest))
                in_cost = costs[source][dest]
                out_cost = costs[dest][source]
                vrp_qubo.add(in_index, in_cost * order_const)
                vrp_qubo.add(out_index, out_cost * order_const)

            # Capacity constraints
            if capacity_const != 0:
                capacity = capacities[vehicle]
                cap_qubo = self.get_capacity_qubo(capacity, start, final)
                vrp_qubo.merge_with(cap_qubo, 1., capacity_const)

            start = final + 1

        return vrp_qubo

    # All vehicles have limit of orders.
    def get_qubo_with_limits(self, vehicles_limits, only_one_const, order_const, capacity_const):
        steps = 0
        for limit in vehicles_limits:
            steps += limit
        dests_num = len(self.dests)

        fake_const = len(self.weights)
        fake_dests_range = range(fake_const, fake_const + steps - dests_num)

        # Adding fake orders
        for fake_dest in fake_dests_range:
            self.costs[fake_dest] = dict()
            for dest in self.dests:
                self.costs[dest][fake_dest] = 0
                self.costs[fake_dest][dest] = only_one_const
            self.costs[self.source][fake_dest] = 0
            self.costs[fake_dest][self.source] = only_one_const
        for fake_dest in fake_dests_range:
            self.dests.append(fake_dest)
            self.weights.append(0)
        for (d1, d2) in product(fake_dests_range, fake_dests_range):
            self.costs[d1][d2] = only_one_const

        vrp_qubo = self.get_qubo_with_partition(vehicles_limits, only_one_const, order_const, capacity_const)

        # Removing fake orders.
        for fake_dest in fake_dests_range:
            del self.costs[fake_dest]
            #for dest in self.dests:
            #    del self.costs[dest][fake_dest]
            #del self.costs[self.source][fake_dest]
            #del self.costs[fake_dest][self.source]
        for fake_dest in fake_dests_range:
            self.dests.pop()
            self.weights.pop()

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
                if dest < len(self.weights):
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

