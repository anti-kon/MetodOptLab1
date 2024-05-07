import copy
import math
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from random import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from numba import jit

VERTICES_NUM = 20
WEIGHT_LIMIT = 10
RATING_LIMIT = 10
EDGE_PROBABILITY = 1.0

VISUALIZATION_SEED = 7
NODE_SIZE = 300
EDGE_WIDTH = 2


def draw_graph_with_path(adjacency_matrix, path):
    real_matrix = np.zeros((len(adjacency_matrix), len(adjacency_matrix)))
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            real_matrix[i][j] = adjacency_matrix[i][j][0]

    G = nx.from_numpy_array(real_matrix, create_using=nx.MultiDiGraph())

    for first_node_index in range(0, len(real_matrix)):
        for second_node_index in range(0, len(real_matrix)):
            if real_matrix[first_node_index][second_node_index] == np.inf:
                G.remove_edge(first_node_index, second_node_index)

    edges_path = list(zip(path, path[1:]))
    edges_path_reversed = [(y, x) for (x, y) in edges_path]
    edges_path = edges_path + edges_path_reversed
    edge_colors = ['k' if not edge in edges_path else 'r' for edge in G.edges()]
    nodecol = ['k' if not node in path else 'r' for node in G.nodes()]

    pos = nx.shell_layout(G)  # positions for all nodes - seed for reproducibility

    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.01] * 4)]

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=nodecol)
    nx.draw_networkx_labels(G, pos, font_color="w")
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, connectionstyle=connectionstyle, width=EDGE_WIDTH,
    )
    labels = {
        tuple(edge): f"{attrs['weight']}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    # nx.draw_networkx_edge_labels(
    #     G,
    #     pos,
    #     labels,
    #     connectionstyle=connectionstyle,
    #     bbox={"color": 'w', "alpha": 1}
    # )

    ax = plt.gca()
    ax.margins(0.001)
    plt.axis("off")
    plt.show()


def generate_graph_matrix(vertices_num, weight_limit, rating_limit, edge_probability):
    matrix = np.zeros((vertices_num, vertices_num, 2))
    for row_index in range(0, vertices_num):
        for column_index in range(0, vertices_num):
            if row_index != column_index and random() < edge_probability:
                matrix[column_index][row_index][0] = math.floor(random() * weight_limit + 1)
    nodes_weight = np.floor(np.random.rand(vertices_num) * rating_limit + 1)
    for row_index in range(0, vertices_num):
        for column_index in range(0, vertices_num):
            if row_index != column_index and matrix[column_index][row_index][0] != 0.0:
                matrix[column_index][row_index][1] = (nodes_weight[row_index])
    return matrix


def ant_colony_optimization(adjacency_matrix, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = len(adjacency_matrix)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf

    for iteration in range(n_iterations):
        paths = []
        path_lengths = []

        for ant in range(n_ants):
            visited = [False] * n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0
            rating = 0

            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))

                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = pheromone[current_point, unvisited_point] ** alpha / (
                        adjacency_matrix[current_point][unvisited_point][0]) ** beta

                probabilities /= np.sum(probabilities)

                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += adjacency_matrix[current_point][next_point][0]
                rating += adjacency_matrix[current_point][next_point][1]
                visited[next_point] = True
                current_point = next_point

            paths.append(path)
            path_lengths.append(path_length)

            if path_length < best_path_length and rating < 1000:
                best_path = path
                best_path_length = path_length

        pheromone *= evaporation_rate

        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points - 1):
                pheromone[path[i], path[i + 1]] += Q / path_length
            pheromone[path[-1], path[0]] += Q / path_length

    return best_path, best_path_length


def solve_tsp_dynamic_programming(
        distance_matrix: np.ndarray,
        maxsize: Optional[int] = None,
) -> Tuple[List, float]:
    N = frozenset(range(1, distance_matrix.shape[0]))
    memo: Dict[Tuple, int] = {}

    @lru_cache(maxsize=maxsize)
    def dist(ni: int, N: frozenset) -> [float, float]:
        if not N:
            return distance_matrix[ni, 0]

        costs = []
        ratings = []
        for nj in N:
            local = dist(nj, N.difference({nj}))
            costs += [(nj, distance_matrix[ni, nj, 0] + local[0])]
            ratings += [(nj, distance_matrix[ni, nj, 1] + local[1])]
        available = set()
        for i in ratings:
            if i[1] < 100:
                available.add(i[0])
        updated_cost = []
        for i in costs:
            if i[0] in available:
                updated_cost.append(i)
        nmin, min_cost = min(updated_cost, key=lambda x: x[1])
        updated_rating = []
        for i in ratings:
            if i[0] == nmin:
                updated_rating.append(i)
        nmin, min_rating = min(updated_rating, key=lambda x: x[1])
        memo[(ni, N)] = nmin

        return [min_cost, min_rating]

    best_distance = dist(0, N)

    ni = 0
    solution = [0]

    while N:
        ni = memo[(ni, N)]
        solution.append(ni)
        N = N.difference({ni})

    return solution, best_distance


# for i in range(0, len(adjacency_matrix)):
#     for j in range(0, len(adjacency_matrix)):
#         print(adjacency_matrix[i][j][1], end=' ')
#     print()
time_history = [[], []]
node_num = 2
while node_num < 21:
    try:
        history = [[], []]
        iteration = 0
        while iteration < 20:
            try:
                print(node_num, iteration)
                adjacency_matrix = generate_graph_matrix(node_num, WEIGHT_LIMIT, RATING_LIMIT, EDGE_PROBABILITY)

                ant_start = time.time()
                path, total_cost = ant_colony_optimization(adjacency_matrix, n_ants=10, n_iterations=100, alpha=1,
                                                           beta=1,
                                                           evaporation_rate=0.5, Q=1)
                ant_finish = time.time()
                total_cost += adjacency_matrix[path[-1]][path[0]][0]
                path = path + [path[0]]

                tsp_start = time.time()
                path, total_cost = solve_tsp_dynamic_programming(adjacency_matrix)
                tsp_finish = time.time()
                history[0].append((ant_finish - ant_start) * 1000)
                history[1].append((tsp_finish - tsp_start) * 1000)
                iteration += 1
            except:
                continue
        print(node_num)
        print(np.array(history))
        print(sum(history[0]) / len(history[0]))
        time_history[0].append(sum(history[0]) / len(history[0]))
        time_history[1].append(sum(history[1]) / len(history[1]))
        node_num += 1
    except:
        print("!")
        continue
print(time_history)
plt.clf()
plt.grid(True)
x = range(2, len(time_history[0]) + 2)
plt.plot(x, time_history[0], label='ant')
plt.plot(x, time_history[1], label='dynamic')
plt.legend()
plt.show()