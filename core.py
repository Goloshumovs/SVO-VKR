# core.py
from collections import deque
import networkx as nx
import numpy as np


class Dinic:
    def __init__(self, n, edges):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.capacity = {}
        self.flow = {}

        for u, v, cap in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
            self.capacity[(u, v)] = cap
            self.capacity[(v, u)] = 0
            self.flow[(u, v)] = 0
            self.flow[(v, u)] = 0

    def bfs(self, s, t, level):
        for i in range(self.n):
            level[i] = -1
        level[s] = 0
        q = deque([s])

        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if level[v] < 0 and self.capacity[(u, v)] - self.flow[(u, v)] > 0:
                    level[v] = level[u] + 1
                    q.append(v)

        return level[t] >= 0

    def dfs(self, u, t, f, level, it):
        if u == t:
            return f

        for i in range(it[u], len(self.adj[u])):
            it[u] = i
            v = self.adj[u][i]

            if level[v] == level[u] + 1 and self.capacity[(u, v)] - self.flow[(u, v)] > 0:
                pushed = self.dfs(v, t, min(f, self.capacity[(u, v)] - self.flow[(u, v)]), level, it)
                if pushed > 0:
                    self.flow[(u, v)] += pushed
                    self.flow[(v, u)] -= pushed
                    return pushed

        return 0

    def max_flow(self, s, t):
        flow = 0
        level = [0] * self.n

        while self.bfs(s, t, level):
            it = [0] * self.n
            while True:
                pushed = self.dfs(s, t, float('inf'), level, it)
                if pushed == 0:
                    break
                flow += pushed

        flow_dict = {}
        for u in range(self.n):
            flow_dict[u] = {}
            for v in self.adj[u]:
                flow_dict[u][v] = self.flow[(u, v)] if self.flow[(u, v)] > 0 else 0

        return flow, flow_dict


def networkx_to_dinic(graph, source, sink):
    node_to_idx = {node: i for i, node in enumerate(graph.nodes())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    n = len(graph.nodes())

    edges = []
    for u, v, data in graph.edges(data=True):
        cap = data.get('capacity', 0)
        if cap > 0:
            edges.append((node_to_idx[u], node_to_idx[v], cap))

    dinic = Dinic(n, edges)
    flow_value, flow_dict_idx = dinic.max_flow(node_to_idx[source], node_to_idx[sink])

    flow_dict = {}
    for u in graph.nodes():
        flow_dict[u] = {}
        for v in graph.nodes():
            flow_dict[u][v] = 0

    for u_idx, v_dict in flow_dict_idx.items():
        u = idx_to_node[u_idx]
        for v_idx, val in v_dict.items():
            if val > 0:
                v = idx_to_node[v_idx]
                flow_dict[u][v] = val

    return flow_value, flow_dict


class CoreMixin:
    """Ядро: кэш, построение потокового графа, оценка решений."""

    def create_flow_graph(self, original_graph, vertex_capacities):
        flow_graph = nx.DiGraph()
        for vertex in vertex_capacities:
            flow_graph.add_node(f"{vertex}_in")
            flow_graph.add_node(f"{vertex}_out")
            flow_graph.add_edge(f"{vertex}_in", f"{vertex}_out", capacity=vertex_capacities[vertex])
        for start, end in original_graph.edges():
            flow_graph.add_edge(f"{start}_out", f"{end}_in", capacity=float('inf'))
        return flow_graph

    def compute_max_flow(self, graph, source, sink, use_cache=False):
        if use_cache:
            key = tuple(sorted([(u, v, data['capacity']) for u, v, data in graph.edges(data=True)]))
            if key in self.cache:
                return self.cache[key]

        result = networkx_to_dinic(graph, source, sink)

        if use_cache:
            key = tuple(sorted([(u, v, data['capacity']) for u, v, data in graph.edges(data=True)]))
            self.cache[key] = result

        return result

    def clear_cache(self):
        self.cache = {}
        self.elite_set = []

    def evaluate_solution(self, resources, vertex_capacities, graph, count):
        current_capacities = dict(vertex_capacities)
        for i in range(1, count + 1):
            if current_capacities[i] != float('inf'):
                current_capacities[i] = vertex_capacities[i] + 2 * np.sqrt(resources[i])

        flow_graph = self.create_flow_graph(graph, current_capacities)
        destination_vertex = f"{count+1}_in"
        flow_value, flow_dict = self.compute_max_flow(flow_graph, '0_in', destination_vertex, use_cache=False)

        return flow_value, flow_dict, current_capacities