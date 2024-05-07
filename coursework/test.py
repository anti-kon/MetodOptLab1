# from numba import jit, cuda, typed, int32, int64
# import numpy as np
# # to measure exec time
# from timeit import default_timer as timer
#
#
# # normal function to run on cpu
# def func(a):
#     for i in range(10000000):
#         a[i] += 1
#
#
# # function optimized to run on gpu
# @jit(target_backend='cuda')
# def func2(a):
#     for i in range(10000000):
#         a[i] += 1
#
#
# if __name__ == "__main__":
#     n = 10000000
#     a = np.ones(n, dtype=np.float64)
#
#     start = timer()
#     func(a)
#     print("without GPU:", timer() - start)
#
#     start = timer()
#     func2(a)
#     print("with GPU:", timer() - start)
from numba import jit, cuda, typed, int32, int64, njit, float64, typeof, types
import numpy as np
from numba.experimental import jitclass

#
# @jitclass([('visited_nodes', types.Set(int64, reflected=True)), ('current_node', int64)])
# class PathKey:
#     def __init__(self, visited_nodes, current_node):
#         self.visited_nodes = visited_nodes
#         self.current_node = current_node
#
#     def __eq__(self, other):
#         if isinstance(other, PathKey):
#             return self.current_node == other.current_node and self.visited_nodes == other.visited_nodes
#
#     def __hash__(self):
#         return hash(str(self.visited_nodes) + str(self.current_node))
#
#     def __str__(self):
#         return f"<PathKey({str(self.visited_nodes)},{self.current_node})>"
#
#     @property
#     def get_length(self):
#         return len(self.visited_nodes)


@jitclass([('path', int32[::1]), ('sum', float64), ])
class PathValue:
    def __init__(self, path, sum):
        self.path = path
        self.sum = sum


# k_v = (typeof(PathKey({0}, 0)), typeof(PathValue(np.array([0, 2]), 0.0)))
#
#
# @njit
# def make_dict():
#     return typed.Dict.empty(*k_v)
#

def generate(vertices_num, iterations, adjacency_matrix, end_node):
    @jit(nopython=True)
    def aux(memoization, current_node_index, visited_nodes, path, sum):
        return (vertices_num, iterations, adjacency_matrix, end_node,
                memoization, current_node_index, visited_nodes, path, sum)

    return aux


class MyClass(object):
    def __init__(self, adjacency_matrix, start_node):
        self.adjacency_matrix = adjacency_matrix
        self.start_node = start_node
        self.vertices_num = len(adjacency_matrix)

        self.end_node = start_node
        self.memoization = typed.Dict()
        self.memoization[({0}, 0)] = PathValue(np.array([0]), 0.0)

        self.shortest_path = typed.List.empty_list(int32)
        self.min_path_cost = np.inf
        self.iterations = 0

    def _gen_method(self):
        return generate(self.vertices_num, self.iterations, self.adjacency_matrix, self.end_node)

    def mymethod(self, current_node_index, visited_nodes, path, sum):
        # self.memoization[PathKey({0, 1}, 0)] = PathValue(np.array([1]), 0.0)
        return self._gen_method()(self.memoization, current_node_index, visited_nodes, path, sum)

memoization = typed.Dict()
memoization[({0}, 0)] = PathValue(np.array([0]), 0.0)
# inst = MyClass(np.array([[1, 2], [2, 3]]), 0)
print(typeof(memoization))
