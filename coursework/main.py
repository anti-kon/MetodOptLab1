from numba import jit, cuda, types, typed, typeof, njit
from numba.core import cgutils
import time
import multiprocessing as mp
from random import random
import itertools as it
import matplotlib.pyplot as plt
from numba import int32, int64, float32, float64
import networkx as nx
import numpy as np
from numba.experimental import jitclass
from numba.extending import (typeof_impl, type_callable, models, register_model, make_attribute_wrapper, lower_builtin,
                             unbox, NativeValue, box, overload)

VERTICES_NUM = 10
WEIGHT_LIMIT = 10
EDGE_PROBABILITY = 1.0

VISUALIZATION_SEED = 7
NODE_SIZE = 300
EDGE_WIDTH = 2


@jitclass([('visited_nodes', types.Set(int64, reflected=True)), ('current_node', int64)])
class PathKey:
    def __init__(self, visited_nodes, current_node):
        self.visited_nodes = visited_nodes
        self.current_node = current_node

    def __eq__(self, other):
        if isinstance(other, PathKey):
            return self.current_node == other.current_node and self.visited_nodes == other.visited_nodes

    def __hash__(self):
        return hash(str(self.visited_nodes) + str(self.current_node))

    def __str__(self):
        return f"<PathKey({str(self.visited_nodes)},{self.current_node})>"

    @property
    def get_length(self):
        return len(self.visited_nodes)


#
#
# class PathKeyType(types.Type):
#     def __init__(self):
#         super(PathKeyType, self).__init__(name='PathKey')
#
#
# path_key_type = PathKeyType()
#
#
# @typeof_impl.register(PathKey)
# def typeof_index(val, c):
#     return path_key_type
#
#
# @type_callable(PathKey)
# def type_interval(context):
#     def typer(visited_nodes, current_node):
#         if isinstance(visited_nodes, types.Set) and isinstance(current_node, types.Integer):
#             return path_key_type
#
#     return typer
#
#
# @register_model(PathKeyType)
# class PathKeyModel(models.StructModel):
#     def __init__(self, dmm, fe_type):
#         members = [
#             ('visited_nodes', types.int64),
#             ('current_node', types.int64),
#             ('hash', types.int64)
#         ]
#         models.StructModel.__init__(self, dmm, fe_type, members)
#
#
# make_attribute_wrapper(PathKeyType, 'visited_nodes', 'visited_nodes')
# make_attribute_wrapper(PathKeyType, 'current_node', 'current_node')
# make_attribute_wrapper(PathKeyType, 'hash', 'hash')
#
#
# @lower_builtin(PathKey, types.Integer, types.Integer)
# def impl_interval(context, builder, sig, args):
#     typ = sig.return_type
#     visited_nodes, current_node = args
#     path_key = cgutils.create_struct_proxy(typ)(context, builder)
#     path_key.visited_nodes = visited_nodes
#     path_key.current_node = current_node
#     return path_key._getvalue()
#
#
# @unbox(PathKeyType)
# def unbox_interval(typ, obj, c):
#     """
#     Convert a Interval object to a native interval structure.
#     """
#     visited_nodes_obj = c.pyapi.object_getattr_string(obj, "visited_nodes")
#     current_node_obj = c.pyapi.object_getattr_string(obj, "current_node")
#     hash_obj = c.pyapi.object_getattr_string(obj, "hash")
#     path_key = cgutils.create_struct_proxy(typ)(c.context, c.builder)
#     path_key.visited_nodes = c.pyapi.float_as_double(visited_nodes_obj)
#     path_key.current_node = c.pyapi.float_as_double(current_node_obj)
#     path_key.hash = c.pyapi.float_as_double(hash_obj)
#     c.pyapi.decref(visited_nodes_obj)
#     c.pyapi.decref(current_node_obj)
#     c.pyapi.decref(hash_obj)
#     is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
#     return NativeValue(path_key._getvalue(), is_error=is_error)
#
#
# @box(PathKeyType)
# def box_interval(typ, val, c):
#     """
#     Convert a native interval structure to an Interval object.
#     """
#     path_key = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
#     visited_nodes_obj = c.pyapi.float_from_double(path_key.visited_nodes)
#     current_node_obj = c.pyapi.float_from_double(path_key.current_node)
#     hash_obj = c.pyapi.float_from_double(path_key.hash)
#     class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(PathKey))
#     res = c.pyapi.call_function_objargs(class_obj, (visited_nodes_obj, current_node_obj, hash_obj))
#     c.pyapi.decref(visited_nodes_obj)
#     c.pyapi.decref(current_node_obj)
#     c.pyapi.decref(hash_obj)
#     c.pyapi.decref(class_obj)
#     return res

@jitclass([('path', int32[::1]), ('sum', float64), ])
class PathValue:
    def __init__(self, path, sum):
        self.path = path
        self.sum = sum


#
# class PathValueType(types.Type):
#     def __init__(self):
#         super(PathValueType, self).__init__(name='PathValue')
#
#
# path_value_type = PathValueType()
#
#
# @typeof_impl.register(PathValue)
# def typeof_index(val, c):
#     return path_value_type
#
#
# @type_callable(PathValue)
# def type_interval(context):
#     def typer(path, sum):
#         if isinstance(path, types.Array) and isinstance(sum, types.Float):
#             return path_key_type
#
#     return typer
#
#
# @register_model(PathValueType)
# class PathValueModel(models.StructModel):
#     def __init__(self, dmm, fe_type):
#         members = [
#             ('path', types.int32[::1]),
#             ('sum', types.float64),
#         ]
#         models.StructModel.__init__(self, dmm, fe_type, members)
#
#
# make_attribute_wrapper(PathValueType, 'path', 'path')
# make_attribute_wrapper(PathValueType, 'sum', 'sum')
#
#
# @lower_builtin(PathValue, types.Array, types.Float)
# def impl_interval(context, builder, sig, args):
#     typ = sig.return_type
#     path, sum = args
#     path_value = cgutils.create_struct_proxy(typ)(context, builder)
#     path_value.path = path
#     path_value.sum = sum
#     return path_value._getvalue()
#
#
# @unbox(PathValueType)
# def unbox_interval(typ, obj, c):
#     path_obj = c.pyapi.object_getattr_string(obj, "path")
#     sum_obj = c.pyapi.object_getattr_string(obj, "sum")
#     path_value = cgutils.create_struct_proxy(typ)(c.context, c.builder)
#     path_value.path = c.pyapi.(path_obj)
#     path_value.sum = c.pyapi.float_as_double(sum_obj)
#     c.pyapi.decref(path_obj)
#     c.pyapi.decref(sum_obj)
#     is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
#     return NativeValue(path_value._getvalue(), is_error=is_error)
#
#
# @box(PathValueType)
# def box_interval(typ, val, c):
#     path_value = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
#     path_obj = c.pyapi.float_from_double(path_value.path)
#     sum_obj = c.pyapi.float_from_double(path_value.sum)
#     class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(PathValue))
#     res = c.pyapi.call_function_objargs(class_obj, (path_obj, sum_obj))
#     c.pyapi.decref(path_obj)
#     c.pyapi.decref(sum_obj)
#     c.pyapi.decref(class_obj)
#     return res


kv_ty = (int64, int64,)





@jitclass([('adjacency_matrix', float64[:, ::1]),
           ('start_node', int64),
           ('vertices_num', int64),
           ('end_node', int64),
           ('memoization', types.DictType(*kv_ty)),
           ('shortest_path', types.ListType(types.int32)),
           ('min_path_cost', float64),
           ('iterations', int64)])
class DynamicSolver:
    def __init__(self, adjacency_matrix, start_node):
        self.adjacency_matrix = adjacency_matrix
        self.start_node = start_node
        self.vertices_num = len(adjacency_matrix)

        self.end_node = start_node
        self.memoization = typed.Dict.empty(*kv_ty)

        self.shortest_path = typed.List.empty_list(types.int32)
        self.min_path_cost = np.inf
        self.iterations = 0

    def solve(self):
        self.visit_neighboring_nodes(self.start_node, {self.start_node}, [self.start_node], 0)
        for key, value in self.memoization.items():
            if key.get_length == self.vertices_num and value.path[-1] == self.end_node:
                self.shortest_path = value.path
                self.min_path_cost = value.sum
        print("done")

    def visit_neighboring_nodes(self, current_node_index, visited_nodes, path, sum):
        for neighboring_index in range(0, self.vertices_num):
            self.iterations += 1
            if (self.adjacency_matrix[current_node_index][neighboring_index] != np.inf and
                    neighboring_index != current_node_index):
                local_sum = sum + self.adjacency_matrix[current_node_index][neighboring_index]
                visited_nodes = visited_nodes | {neighboring_index}
                local_key = PathKey(visited_nodes, neighboring_index)
                if local_key not in self.memoization or self.memoization[local_key].sum > local_sum:
                    self.memoization[local_key] = PathValue(np.concatenate((path, [neighboring_index])), local_sum)
                    if len(visited_nodes) != self.vertices_num or path[-1] != self.end_node:
                        self.visit_neighboring_nodes(neighboring_index, visited_nodes | {neighboring_index},
                                                     path + [neighboring_index],
                                                     local_sum)


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
            if adjacency_matrix[first_node_index][second_node_index] == np.inf:
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
    fo_pow_y = [4 ** element for element in x_vector]
    fi_pow_y = [5 ** element for element in x_vector]
    th_pow_y = [3 ** element for element in x_vector]
    plt.plot(x_vector, fi_pow_y, label='5 ** n')
    plt.plot(x_vector, fo_pow_y, label='4 ** n')
    plt.plot(x_vector, th_pow_y, label='3 ** n')
    plt.plot(x_vector, y_vector, label='our method')
    plt.legend()
    plt.show()


def make_pr(node_num):
    # a = dict()
    # x = ({1, 2}, 0)
    # y = ({1, 2}, 0)
    # a[str(x)] = 2
    # print(a)
    # a[str(y)] = 3
    # print(a)
    adjacency_matrix = generate_graph_matrix(node_num, WEIGHT_LIMIT, EDGE_PROBABILITY)

    for first_node_index in range(0, len(adjacency_matrix)):
        for second_node_index in range(0, len(adjacency_matrix)):
            if adjacency_matrix[first_node_index][second_node_index] == 0:
                adjacency_matrix[first_node_index][second_node_index] = np.inf

    # Dynamic algorithm
    result = DynamicSolver(adjacency_matrix, 0)
    start = time.time()
    result.solve()
    finish = time.time()
    # return [result.iterations, (finish - start) * 1000]


# @njit
# def get_d():
#     return typed.Dict.empty(key_type=int64, value_type=path_value_type, )


if __name__ == '__main__':
    start = time.time()
    make_pr(500)
    finish = time.time()
    print((finish - start) * 1000)

    # pool = mp.Pool(8)
    #
    # start = time.time()
    # async_result = []
    # for i in range(50):
    #     async_result.append(pool.map_async(make_pr, range(2, 11)))
    # finish = time.time()
    # for i in range(0, len(async_result)):
    #     print(i, async_result[i].get())
    #
    # dynamic_method_result_statistic = [0] * len(async_result[-1].get())
    # for variation in async_result:
    #     for result_index in range(0, len(variation.get())):
    #         dynamic_method_result_statistic[result_index] = (dynamic_method_result_statistic[result_index] +
    #                                                          variation.get()[result_index][0]) / 2
    # print(dynamic_method_result_statistic)
    # draw_graphic(range(2, len(dynamic_method_result_statistic) + 2), dynamic_method_result_statistic)

    # # Christofides
    # path = nx.algorithms.approximation.traveling_salesman_problem(G, weight="weight", nodes=None, cycle=True,
    #                                                               method=None)
    # christofides_sum = 0
    # for v, u in edges_path:
    #     christofides_sum += adjacency_matrix[v][u]
