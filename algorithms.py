# algorithms.py
import numpy as np
import random


class AlgorithmsMixin:
    """Все алгоритмы распределения ресурсов."""

    # ==================== ЖАДНЫЙ АЛГОРИТМ ====================
    def greedy_algorithm(self, graph, count, vertex_capacities, budget, flow_value, proportions):
        resources = np.zeros(count + 1)
        current_capacities = dict(vertex_capacities)
        remaining_budget = budget
        step = self.step
        destination_vertex = f"{count+1}_in"

        while remaining_budget >= step:
            best_increase = -1
            best_vertex = -1
            best_new_capacity = 0
            best_flow_value = 0

            for i in range(1, count + 1):
                if current_capacities[i] == float('inf'):
                    continue
                if proportions[i][1] > 0:
                    continue

                current_investment = resources[i]
                new_investment = current_investment + step
                new_capacity = vertex_capacities[i] + 2 * np.sqrt(new_investment)

                test_capacities = dict(current_capacities)
                test_capacities[i] = new_capacity

                test_flow_graph = self.create_flow_graph(graph, test_capacities)
                test_flow_value, _ = self.compute_max_flow(test_flow_graph, '0_in', destination_vertex, use_cache=True)

                increase = test_flow_value - flow_value

                if increase > best_increase:
                    best_increase = increase
                    best_vertex = i
                    best_new_capacity = new_capacity
                    best_flow_value = test_flow_value

            if best_vertex > 0:
                resources[best_vertex] += step
                current_capacities[best_vertex] = best_new_capacity
                flow_value = best_flow_value
                remaining_budget -= step

                flow_graph = self.create_flow_graph(graph, current_capacities)
                test_flow_value, test_flow_dict = self.compute_max_flow(flow_graph, '0_in', destination_vertex, use_cache=False)
                new_loads = {}
                for start in test_flow_dict:
                    node_id_str = start.split('_')[0]
                    try:
                        node_id = int(node_id_str)
                        new_loads[node_id] = sum(test_flow_dict[start].values())
                        new_loads[node_id] = np.nan_to_num(new_loads[node_id])
                    except ValueError:
                        continue
                for node in range(1, count + 1):
                    proportions[node] = [new_loads.get(node, 0), current_capacities.get(node, 0) - new_loads.get(node, 0)]
            else:
                break

        if remaining_budget > 0:
            for i in range(1, count + 1):
                if current_capacities[i] != float('inf') and proportions[i][1] == 0:
                    resources[i] += remaining_budget
                    current_capacities[i] = vertex_capacities[i] + 2 * np.sqrt(resources[i])
                    remaining_budget = 0
                    break
            else:
                for i in range(1, count + 1):
                    if current_capacities[i] != float('inf'):
                        resources[i] += remaining_budget
                        current_capacities[i] = vertex_capacities[i] + 2 * np.sqrt(resources[i])
                        remaining_budget = 0
                        break

        return resources, current_capacities, flow_value, proportions

    # ==================== REGRET-ВЫБОР ====================
    def regret_algorithm(self, graph, count, vertex_capacities, budget, flow_value, proportions, regret_weight=0.5):
        resources = np.zeros(count + 1)
        current_capacities = dict(vertex_capacities)
        remaining_budget = budget
        step = self.step
        destination_vertex = f"{count+1}_in"

        if budget > 100000:
            step = 100
        elif budget > 10000:
            step = 50

        while remaining_budget >= step:
            candidates = []

            for i in range(1, count + 1):
                if current_capacities[i] == float('inf'):
                    continue
                if proportions[i][1] > 0:
                    continue

                current_investment = resources[i]
                new_investment = current_investment + step
                new_capacity = vertex_capacities[i] + 2 * np.sqrt(new_investment)

                test_capacities = dict(current_capacities)
                test_capacities[i] = new_capacity

                test_flow_graph = self.create_flow_graph(graph, test_capacities)
                test_flow_value, _ = self.compute_max_flow(test_flow_graph, '0_in', destination_vertex, use_cache=True)

                increase = test_flow_value - flow_value
                candidates.append((i, increase, new_capacity, new_investment))

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[1])
            best_gain = candidates[0][1]

            for idx, (vertex, gain, new_cap, inv) in enumerate(candidates):
                regret = best_gain - gain
                candidates[idx] = (vertex, gain + regret_weight * regret, new_cap, inv, regret)

            candidates.sort(key=lambda x: -x[1])

            best_vertex = candidates[0][0]
            best_new_capacity = candidates[0][2]

            resources[best_vertex] += step
            current_capacities[best_vertex] = best_new_capacity
            flow_value += candidates[0][1]
            remaining_budget -= step

            flow_graph = self.create_flow_graph(graph, current_capacities)
            test_flow_value, test_flow_dict = self.compute_max_flow(flow_graph, '0_in', destination_vertex, use_cache=False)
            new_loads = {}
            for start in test_flow_dict:
                node_id_str = start.split('_')[0]
                try:
                    node_id = int(node_id_str)
                    new_loads[node_id] = sum(test_flow_dict[start].values())
                    new_loads[node_id] = np.nan_to_num(new_loads[node_id])
                except ValueError:
                    continue
            for node in range(1, count + 1):
                proportions[node] = [new_loads.get(node, 0), current_capacities.get(node, 0) - new_loads.get(node, 0)]

        if remaining_budget > 0:
            for i in range(1, count + 1):
                if current_capacities[i] != float('inf') and proportions[i][1] == 0:
                    resources[i] += remaining_budget
                    current_capacities[i] = vertex_capacities[i] + 2 * np.sqrt(resources[i])
                    remaining_budget = 0
                    break
            else:
                for i in range(1, count + 1):
                    if current_capacities[i] != float('inf'):
                        resources[i] += remaining_budget
                        current_capacities[i] = vertex_capacities[i] + 2 * np.sqrt(resources[i])
                        remaining_budget = 0
                        break

        return resources, current_capacities, flow_value, proportions

    # ==================== GRASP ====================
    def grasp_construction(self, graph, count, vertex_capacities, budget, flow_value, proportions, alpha=None):
        if alpha is None:
            alpha = self.grasp_alpha

        resources = np.zeros(count + 1)
        current_capacities = dict(vertex_capacities)
        remaining_budget = budget
        step = self.step
        destination_vertex = f"{count+1}_in"

        if budget > 100000:
            step = 100
        elif budget > 10000:
            step = 50

        while remaining_budget >= step:
            candidates = []

            for i in range(1, count + 1):
                if current_capacities[i] == float('inf'):
                    continue
                if proportions[i][1] > 0:
                    continue

                current_investment = resources[i]
                new_investment = current_investment + step
                new_capacity = vertex_capacities[i] + 2 * np.sqrt(new_investment)

                test_capacities = dict(current_capacities)
                test_capacities[i] = new_capacity

                test_flow_graph = self.create_flow_graph(graph, test_capacities)
                test_flow_value, _ = self.compute_max_flow(test_flow_graph, '0_in', destination_vertex, use_cache=True)

                increase = test_flow_value - flow_value
                candidates.append((i, increase, new_capacity))

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[1])

            best_gain = candidates[0][1]
            worst_gain = candidates[-1][1]

            if best_gain == worst_gain:
                threshold = best_gain
            else:
                threshold = best_gain - alpha * (best_gain - worst_gain)

            rcl = [c for c in candidates if c[1] >= threshold]

            selected = random.choice(rcl)
            best_vertex = selected[0]
            best_new_capacity = selected[2]

            resources[best_vertex] += step
            current_capacities[best_vertex] = best_new_capacity
            flow_value += selected[1]
            remaining_budget -= step

            flow_graph = self.create_flow_graph(graph, current_capacities)
            test_flow_value, test_flow_dict = self.compute_max_flow(flow_graph, '0_in', destination_vertex, use_cache=False)
            new_loads = {}
            for start in test_flow_dict:
                node_id_str = start.split('_')[0]
                try:
                    node_id = int(node_id_str)
                    new_loads[node_id] = sum(test_flow_dict[start].values())
                    new_loads[node_id] = np.nan_to_num(new_loads[node_id])
                except ValueError:
                    continue
            for node in range(1, count + 1):
                proportions[node] = [new_loads.get(node, 0), current_capacities.get(node, 0) - new_loads.get(node, 0)]

        if remaining_budget > 0:
            for i in range(1, count + 1):
                if current_capacities[i] != float('inf') and proportions[i][1] == 0:
                    resources[i] += remaining_budget
                    current_capacities[i] = vertex_capacities[i] + 2 * np.sqrt(resources[i])
                    remaining_budget = 0
                    break
            else:
                for i in range(1, count + 1):
                    if current_capacities[i] != float('inf'):
                        resources[i] += remaining_budget
                        current_capacities[i] = vertex_capacities[i] + 2 * np.sqrt(resources[i])
                        remaining_budget = 0
                        break

        return resources, current_capacities, flow_value, proportions

    # ==================== ЛОКАЛЬНЫЙ ПОИСК ====================
    def local_search_only(self, resources, current_capacities, flow_value, graph, count, vertex_capacities, proportions):
        improved = True
        iterations = 0
        max_iterations = self.ls_iterations
        destination_vertex = f"{count+1}_in"

        while improved and iterations < max_iterations:
            iterations += 1
            improved = False
            best_improvement = 0
            best_move = None

            for i in range(1, count + 1):
                if current_capacities[i] == float('inf') or resources[i] < self.step:
                    continue

                for j in range(1, count + 1):
                    if i == j or current_capacities[j] == float('inf'):
                        continue

                    step = min(self.step, resources[i])
                    if step == 0:
                        continue

                    test_resources = resources.copy()
                    test_resources[i] -= step
                    test_resources[j] += step

                    test_capacities = dict(vertex_capacities)
                    for k in range(1, count + 1):
                        if test_capacities[k] != float('inf'):
                            test_capacities[k] = vertex_capacities[k] + 2 * np.sqrt(test_resources[k])

                    test_flow_graph = self.create_flow_graph(graph, test_capacities)
                    test_flow_value, _ = self.compute_max_flow(test_flow_graph, '0_in', destination_vertex, use_cache=False)

                    increase = test_flow_value - flow_value

                    if increase > best_improvement:
                        best_improvement = increase
                        best_move = (i, j, step, test_resources, test_capacities, test_flow_value)

            if best_move and best_improvement > 0:
                i, j, step, new_resources, new_capacities, new_flow = best_move
                resources = new_resources
                current_capacities = new_capacities
                flow_value = new_flow
                improved = True

                flow_graph = self.create_flow_graph(graph, current_capacities)
                test_flow_value, test_flow_dict = self.compute_max_flow(flow_graph, '0_in', destination_vertex, use_cache=False)
                new_loads = {}
                for start in test_flow_dict:
                    node_id_str = start.split('_')[0]
                    try:
                        node_id = int(node_id_str)
                        new_loads[node_id] = sum(test_flow_dict[start].values())
                        new_loads[node_id] = np.nan_to_num(new_loads[node_id])
                    except ValueError:
                        continue
                for node in range(1, count + 1):
                    proportions[node] = [new_loads.get(node, 0), current_capacities.get(node, 0) - new_loads.get(node, 0)]

        return resources, current_capacities, flow_value, proportions

    # ==================== GRASP + ЛОКАЛЬНЫЙ ПОИСК ====================
    def grasp_with_local_search(self, graph, count, vertex_capacities, budget, flow_value, proportions, alpha=None):
        if alpha is None:
            alpha = self.grasp_alpha

        resources, current_capacities, flow_value, proportions = self.grasp_construction(
            graph, count, vertex_capacities, budget, flow_value, proportions, alpha=alpha
        )

        resources, current_capacities, flow_value, proportions = self.local_search_only(
            resources, current_capacities, flow_value, graph, count, vertex_capacities, proportions
        )

        return resources, current_capacities, flow_value, proportions

    # ==================== REGRET + ЛОКАЛЬНЫЙ ПОИСК ====================
    def regret_with_local_search(self, graph, count, vertex_capacities, budget, flow_value, proportions):
        resources, current_capacities, flow_value, proportions = self.regret_algorithm(
            graph, count, vertex_capacities, budget, flow_value, proportions, regret_weight=0.5
        )

        resources, current_capacities, flow_value, proportions = self.local_search_only(
            resources, current_capacities, flow_value, graph, count, vertex_capacities, proportions
        )

        return resources, current_capacities, flow_value, proportions

    # ==================== PATH RELINKING ====================
    def update_elite_set(self, resources, flow_value):
        for existing_res, existing_flow in self.elite_set:
            if np.array_equal(existing_res, resources):
                return

        self.elite_set.append((resources.copy(), flow_value))
        self.elite_set.sort(key=lambda x: -x[1])

        if len(self.elite_set) > self.elite_size:
            self.elite_set = self.elite_set[:self.elite_size]

    def path_relinking(self, resources1, resources2, graph, count, vertex_capacities):
        diff_indices = []
        for i in range(1, count + 1):
            if abs(resources1[i] - resources2[i]) > 0.001:
                diff_indices.append(i)

        if not diff_indices:
            return resources1, self.evaluate_solution(resources1, vertex_capacities, graph, count)[0]

        diff_indices.sort(key=lambda x: abs(resources1[x] - resources2[x]), reverse=True)

        current = resources1.copy()
        best = resources1.copy()
        best_flow, _, _ = self.evaluate_solution(best, vertex_capacities, graph, count)

        for idx in diff_indices:
            diff = resources2[idx] - current[idx]
            if abs(diff) > 0.001:
                step_size = diff / 2
                current[idx] += step_size

                current_flow, _, _ = self.evaluate_solution(current, vertex_capacities, graph, count)

                if current_flow > best_flow:
                    best = current.copy()
                    best_flow = current_flow

        final_flow, _, _ = self.evaluate_solution(resources2, vertex_capacities, graph, count)
        if final_flow > best_flow:
            best = resources2.copy()
            best_flow = final_flow

        return best, best_flow

    def grasp_with_path_relinking(self, graph, count, vertex_capacities, budget, flow_value, proportions, iterations=5):
        best_resources = None
        best_capacities = None
        best_flow = flow_value
        best_proportions = None

        self.elite_set = []

        for iteration in range(iterations):
            resources, current_capacities, current_flow, proportions_copy = self.grasp_construction(
                graph, count, vertex_capacities, budget, flow_value,
                {k: v.copy() for k, v in proportions.items()}, alpha=self.grasp_alpha
            )

            resources, current_capacities, current_flow, proportions_copy = self.local_search_only(
                resources, current_capacities, current_flow, graph, count, vertex_capacities, proportions_copy
            )

            self.update_elite_set(resources, current_flow)

            if len(self.elite_set) > 1:
                elite_idx = random.randint(0, len(self.elite_set) - 1)
                elite_resources, elite_flow = self.elite_set[elite_idx]

                pr_resources, pr_flow = self.path_relinking(
                    resources, elite_resources, graph, count, vertex_capacities
                )

                if pr_flow > current_flow:
                    resources = pr_resources
                    current_flow = pr_flow
                    _, _, current_capacities = self.evaluate_solution(resources, vertex_capacities, graph, count)

                    flow_graph = self.create_flow_graph(graph, current_capacities)
                    destination_vertex = f"{count+1}_in"
                    test_flow_value, test_flow_dict = self.compute_max_flow(flow_graph, '0_in', destination_vertex, use_cache=False)
                    new_loads = {}
                    for start in test_flow_dict:
                        node_id_str = start.split('_')[0]
                        try:
                            node_id = int(node_id_str)
                            new_loads[node_id] = sum(test_flow_dict[start].values())
                            new_loads[node_id] = np.nan_to_num(new_loads[node_id])
                        except ValueError:
                            continue
                    for node in range(1, count + 1):
                        proportions_copy[node] = [new_loads.get(node, 0), current_capacities.get(node, 0) - new_loads.get(node, 0)]

                    self.update_elite_set(resources, current_flow)

            if current_flow > best_flow:
                best_resources = resources
                best_capacities = current_capacities
                best_flow = current_flow
                best_proportions = proportions_copy

        return best_resources, best_capacities, best_flow, best_proportions

    # ==================== SLS ====================
    def sls_algorithm(self, graph, count, vertex_capacities, budget, flow_value, proportions,
                      k_steps=100, k_neighbors=10, tabu_size=5, mu=0.1, improve_freq=20):
        resources, current_capacities, flow_value, proportions = self.greedy_algorithm(
            graph, count, vertex_capacities, budget, flow_value, proportions
        )

        destination_vertex = f"{count+1}_in"
        tabu_list = []
        best_resources = resources.copy()
        best_flow = flow_value
        best_capacities = current_capacities.copy()

        for step in range(k_steps):
            neighbors = []

            for _ in range(k_neighbors):
                op_type = random.choice(['move', 'swap'])

                if op_type == 'move':
                    from_vertex = random.randint(1, count)
                    to_vertex = random.randint(1, count)

                    if from_vertex != to_vertex and resources[from_vertex] > 0:
                        amount = random.randint(1, min(10, int(resources[from_vertex])))
                        neighbor = resources.copy()
                        neighbor[from_vertex] -= amount
                        neighbor[to_vertex] += amount
                        neighbors.append(('move', from_vertex, to_vertex, amount, neighbor))
                else:
                    v1 = random.randint(1, count)
                    v2 = random.randint(1, count)

                    if v1 != v2 and resources[v1] > 0 and resources[v2] > 0:
                        amount1 = random.randint(1, min(10, int(resources[v1])))
                        amount2 = random.randint(1, min(10, int(resources[v2])))
                        neighbor = resources.copy()
                        neighbor[v1] -= amount1
                        neighbor[v1] += amount2
                        neighbor[v2] -= amount2
                        neighbor[v2] += amount1
                        neighbors.append(('swap', v1, v2, amount1, amount2, neighbor))

            best_neighbor = None
            best_neighbor_flow = -1
            best_neighbor_resources = None

            for neighbor_data in neighbors:
                if len(neighbor_data) == 5:
                    _, from_v, to_v, amount, neighbor_res = neighbor_data
                else:
                    _, v1, v2, amount1, amount2, neighbor_res = neighbor_data

                if any(np.array_equal(neighbor_res, tabu_res) for tabu_res, _ in tabu_list):
                    continue

                total_investment = sum(neighbor_res)
                penalty = mu * max(0, total_investment - budget)

                neighbor_flow, _, neighbor_caps = self.evaluate_solution(
                    neighbor_res, vertex_capacities, graph, count
                )

                objective = neighbor_flow - penalty

                if objective > best_neighbor_flow:
                    best_neighbor_flow = objective
                    best_neighbor = neighbor_data
                    best_neighbor_resources = neighbor_res

            if best_neighbor is not None and best_neighbor_flow > flow_value:
                resources = best_neighbor_resources
                flow_value, _, current_capacities = self.evaluate_solution(
                    resources, vertex_capacities, graph, count
                )

                tabu_list.append((resources.copy(), step))
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)

                if flow_value > best_flow:
                    best_resources = resources.copy()
                    best_flow = flow_value
                    best_capacities = current_capacities.copy()

            if step % improve_freq == 0 and step > 0:
                subset_size = random.randint(3, min(7, count))
                subset = random.sample(range(1, count + 1), subset_size)

                total_subset_resources = sum(resources[i] for i in subset)
                for i in subset:
                    resources[i] = total_subset_resources / len(subset)

                for i in subset:
                    resources[i] = round(resources[i])

                new_flow, _, new_caps = self.evaluate_solution(
                    resources, vertex_capacities, graph, count
                )

                if new_flow > best_flow:
                    best_resources = resources.copy()
                    best_flow = new_flow
                    best_capacities = new_caps.copy()

                    flow_value = new_flow
                    current_capacities = new_caps

                    flow_graph = self.create_flow_graph(graph, current_capacities)
                    test_flow_value, test_flow_dict = self.compute_max_flow(flow_graph, '0_in', destination_vertex, use_cache=False)
                    new_loads = {}
                    for start in test_flow_dict:
                        node_id_str = start.split('_')[0]
                        try:
                            node_id = int(node_id_str)
                            new_loads[node_id] = sum(test_flow_dict[start].values())
                            new_loads[node_id] = np.nan_to_num(new_loads[node_id])
                        except ValueError:
                            continue
                    for node in range(1, count + 1):
                        proportions[node] = [new_loads.get(node, 0), current_capacities.get(node, 0) - new_loads.get(node, 0)]

        return best_resources, best_capacities, best_flow, proportions

    # ==================== КОМБИНИРОВАННЫЙ МЕТОД ====================
    def combined_grasp_regret_ls(self, graph, count, vertex_capacities, budget, flow_value, proportions,
                                 alpha=None, regret_prob=0.5, regret_weight=0.5):
        if alpha is None:
            alpha = self.grasp_alpha

        resources = np.zeros(count + 1)
        current_capacities = dict(vertex_capacities)
        remaining_budget = budget
        step = self.step
        destination_vertex = f"{count+1}_in"

        if budget > 100000:
            step = 100
        elif budget > 10000:
            step = 50

        while remaining_budget >= step:
            candidates = []

            for i in range(1, count + 1):
                if current_capacities[i] == float('inf'):
                    continue
                if proportions[i][1] > 0:
                    continue

                current_investment = resources[i]
                new_investment = current_investment + step
                new_capacity = vertex_capacities[i] + 2 * np.sqrt(new_investment)

                test_capacities = dict(current_capacities)
                test_capacities[i] = new_capacity

                test_flow_graph = self.create_flow_graph(graph, test_capacities)
                test_flow_value, _ = self.compute_max_flow(test_flow_graph, '0_in', destination_vertex, use_cache=True)

                increase = test_flow_value - flow_value
                candidates.append((i, increase, new_capacity, new_investment))

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[1])

            best_gain = candidates[0][1]
            for idx, (vertex, gain, new_cap, inv) in enumerate(candidates):
                regret = best_gain - gain
                candidates[idx] = (vertex, gain, new_cap, inv, regret)

            best_gain = candidates[0][1]
            worst_gain = candidates[-1][1]

            if best_gain == worst_gain:
                threshold = best_gain
            else:
                threshold = best_gain - alpha * (best_gain - worst_gain)

            rcl = [c for c in candidates if c[1] >= threshold]

            if random.random() < regret_prob and len(rcl) > 1:
                rcl.sort(key=lambda x: -x[4])
                selected = rcl[0]
            else:
                selected = random.choice(rcl)

            best_vertex = selected[0]
            best_new_capacity = selected[2]

            resources[best_vertex] += step
            current_capacities[best_vertex] = best_new_capacity
            flow_value += selected[1]
            remaining_budget -= step

            flow_graph = self.create_flow_graph(graph, current_capacities)
            test_flow_value, test_flow_dict = self.compute_max_flow(flow_graph, '0_in', destination_vertex, use_cache=False)
            new_loads = {}
            for start in test_flow_dict:
                node_id_str = start.split('_')[0]
                try:
                    node_id = int(node_id_str)
                    new_loads[node_id] = sum(test_flow_dict[start].values())
                    new_loads[node_id] = np.nan_to_num(new_loads[node_id])
                except ValueError:
                    continue
            for node in range(1, count + 1):
                proportions[node] = [new_loads.get(node, 0), current_capacities.get(node, 0) - new_loads.get(node, 0)]

        if remaining_budget > 0:
            for i in range(1, count + 1):
                if current_capacities[i] != float('inf') and proportions[i][1] == 0:
                    resources[i] += remaining_budget
                    current_capacities[i] = vertex_capacities[i] + 2 * np.sqrt(resources[i])
                    remaining_budget = 0
                    break
            else:
                for i in range(1, count + 1):
                    if current_capacities[i] != float('inf'):
                        resources[i] += remaining_budget
                        current_capacities[i] = vertex_capacities[i] + 2 * np.sqrt(resources[i])
                        remaining_budget = 0
                        break

        resources, current_capacities, flow_value, proportions = self.local_search_only(
            resources, current_capacities, flow_value, graph, count, vertex_capacities, proportions
        )

        return resources, current_capacities, flow_value, proportions