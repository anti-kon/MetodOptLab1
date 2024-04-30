import math
import time
from random import random
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

VERTICES_NUM = 10
WEIGHT_LIMIT = 10
EDGE_PROBABILITY = 1.0

VISUALIZATION_SEED = 7
NODE_SIZE = 300
EDGE_WIDTH = 2


class DynamicSolver:
    def __init__(self, adjacency_matrix, start_node):
        self.adjacency_matrix = adjacency_matrix
        self.start_node = start_node
        self.vertices_num = len(adjacency_matrix)

        self.end_node = start_node
        self.memoization = dict()

        self.shortest_path = []
        self.min_path_cost = float('inf')
        self.iterations = 0

    def solve(self):
        self.visit_neighboring_nodes(self.start_node, {self.start_node}, [self.start_node], 0)
        for key, value in self.memoization.items():
            if len(key.visited_nodes) == self.vertices_num and value["path"][-1] == self.end_node:
                print("done")
                self.shortest_path = value["path"]
                self.min_path_cost = value["sum"]

    def visit_neighboring_nodes(self, current_node_index, visited_nodes, path, sum):
        for neighboring_index in range(0, self.vertices_num):
            self.iterations += 1
            if (adjacency_matrix[current_node_index][neighboring_index] != float('inf') and
                    neighboring_index != current_node_index):
                local_sum = sum + adjacency_matrix[current_node_index][neighboring_index]
                local_key = DynamicSolver.PathKey(visited_nodes | {neighboring_index}, neighboring_index)
                if local_key not in self.memoization or self.memoization[local_key].get("sum") > local_sum:
                    self.memoization[local_key] = {"path": path + [neighboring_index], "sum": local_sum}
                    if len(visited_nodes) != self.vertices_num or path[-1] != self.end_node:
                        self.visit_neighboring_nodes(neighboring_index, visited_nodes | {neighboring_index},
                                                     path + [neighboring_index],
                                                     local_sum)

    class PathKey:
        def __init__(self, visited_nodes, current_node):
            self.visited_nodes = visited_nodes
            self.current_node = current_node
            self.hash = self.__hash__()

        def __eq__(self, other):
            if isinstance(other, DynamicSolver.PathKey):
                return self.visited_nodes == other.visited_nodes

        def __hash__(self):
            return hash(str(self.visited_nodes) + str(self.current_node))

        def __repr__(self):
            return f"<PathKey({self.visited_nodes},{self.current_node})>"


def generate_graph_matrix(vertices_num, weight_limit, edge_probability):
    matrix = np.zeros((vertices_num, vertices_num))
    nodes_weight = np.floor(np.random.rand(vertices_num) * weight_limit + 1)
    for row_index in range(0, vertices_num):
        for column_index in range(0, vertices_num):
            if row_index != column_index and random() < edge_probability:
                matrix[column_index][row_index] = nodes_weight[row_index]
    return matrix


def draw_graph_with_path(adjacency_matrix, path):
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.MultiDiGraph())

    for first_node_index in range(0, len(adjacency_matrix)):
        for second_node_index in range(0, len(adjacency_matrix)):
            if adjacency_matrix[first_node_index][second_node_index] == float('inf'):
                G.remove_edge(first_node_index, second_node_index)

    edges_path = list(zip(path, path[1:]))
    edges_path_reversed = [(y, x) for (x, y) in edges_path]
    edges_path = edges_path + edges_path_reversed
    edge_colors = ['k' if not edge in edges_path else 'r' for edge in G.edges()]
    nodecol = ['k' if not node in path else 'r' for node in G.nodes()]

    pos = nx.shell_layout(G)  # positions for all nodes - seed for reproducibility

    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.09] * 4)]

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=nodecol)
    nx.draw_networkx_labels(G, pos, font_color="w")
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, connectionstyle=connectionstyle, width=EDGE_WIDTH,
    )
    labels = {
        tuple(edge): f"{attrs['weight']}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        bbox={"color": 'w', "alpha": 1}
    )

    ax = plt.gca()
    ax.margins(0.001)
    plt.axis("off")
    plt.show()


def draw_graphic(x_vector, y_vector):
    plt.clf()
    plt.grid(True)
    se_pow_y = [4 ** element for element in x_vector]
    th_pow_y = [3 ** element for element in x_vector]
    plt.plot(x_vector, se_pow_y, label='4 ** n')
    plt.plot(x_vector, th_pow_y, label='3 ** n')
    plt.plot(x_vector, y_vector, label='our method')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dynamic_method_result_history = []
    for variation in range(0, 10):
        dynamic_method_result_history.append([])
        for node_num in range(2, 10):
            adjacency_matrix = generate_graph_matrix(node_num, WEIGHT_LIMIT, EDGE_PROBABILITY)

            for first_node_index in range(0, len(adjacency_matrix)):
                for second_node_index in range(0, len(adjacency_matrix)):
                    if adjacency_matrix[first_node_index][second_node_index] == 0:
                        adjacency_matrix[first_node_index][second_node_index] = float('inf')

            # Dynamic algorithm
            result = DynamicSolver(adjacency_matrix, 0)
            start = time.time()
            result.solve()
            finish = time.time()
            dynamic_method_result_history[-1].append([result.iterations, (finish - start) * 1000])
    dynamic_method_result_statistic = [0] * len(dynamic_method_result_history[-1])
    for variation in dynamic_method_result_history:
        for result_index in range(0, len(variation)):
            dynamic_method_result_statistic[result_index] = (dynamic_method_result_statistic[result_index] +
                                                             variation[result_index][0]) / 2
    print(dynamic_method_result_statistic)
    draw_graphic(range(2, len(dynamic_method_result_statistic) + 2), dynamic_method_result_statistic)

    # # Christofides
    # path = nx.algorithms.approximation.traveling_salesman_problem(G, weight="weight", nodes=None, cycle=True,
    #                                                               method=None)
    # christofides_sum = 0
    # for v, u in edges_path:
    #     christofides_sum += adjacency_matrix[v][u]

