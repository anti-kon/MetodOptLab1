import copy

import scipy
import itertools
import math

import numpy as np
import queue

cost_matrix = [[1, 4, 7, 8, 11], [3, 5, 9, 7, 14], [4, 5, 12, 6, 9], [6, 10, 11, 17, 13]]
proposal_vector = [33, 35, 17, 12]
demand_vector = [23, 34, 5, 16, 19]


class Path:
    way = []

    def __init__(self, input_way, new_point):
        input_way.append(new_point)
        self.way = input_way

    def get_possible_moves(self, local_matrix_mask):
        possible_moves = []

        if self.way[-1][2] == -1:
            for row_index in range(0, len(local_matrix_mask)):
                if local_matrix_mask[row_index][self.way[-1][1]] != 0 and row_index != self.way[-1][0]:
                    possible_moves.append([row_index, self.way[-1][1], 1])

        if self.way[-1][2] == 1:
            for column_index in range(0, len(local_matrix_mask[0])):
                if local_matrix_mask[self.way[-1][0]][column_index] != 0 and column_index != self.way[-1][1]:
                    possible_moves.append([self.way[-1][0], column_index, -1])

        return_flag = False
        if len(self.way) > 3 and (
                (self.way[-1][2] == -1 and self.way[0][1] == self.way[-1][1]) or (
                self.way[-1][2] == 1 and self.way[0][0] == self.way[-1][0])):
            return_flag = True
            return return_flag, []

        for possible_move in possible_moves:
            for way_point in range(1, len(self.way)):
                if self.way[way_point][0] == possible_move[0] and self.way[way_point][1] == possible_move[1]:
                    possible_moves.remove(possible_move)

        return return_flag, possible_moves


def sum_balance_check(first_vector, second_vector):
    first_sum = 0
    for i in range(0, len(first_vector)):
        first_sum += first_vector[i]
    second_sum = 0
    for i in range(0, len(second_vector)):
        second_sum += second_vector[i]
    return first_sum - second_sum


def get_used_mask(extended_matrix):
    matrix_mask = []

    for i in range(0, len(extended_matrix) - 1):
        matrix_mask.append([0] * (len(extended_matrix[i]) - 1))
        for j in range(0, (len(extended_matrix[i]) - 1)):
            if extended_matrix[i][j] != 0:
                matrix_mask[i][j] = 1

    return matrix_mask


def northwest_method(local_cost_matrix, local_proposal_vector, local_demand_vector):
    local_path_matrix = []
    m = len(local_cost_matrix[0]) + 1
    for i in range(0, (len(local_cost_matrix) + 1)):
        local_path_matrix.append([0] * m)
    for i in range(0, len(local_proposal_vector)):
        local_path_matrix[i][-1] = local_proposal_vector[i]
    for i in range(0, len(local_demand_vector)):
        local_path_matrix[-1][i] = local_demand_vector[i]

    local_matrix_mask = []
    for i in range(0, len(local_cost_matrix)):
        local_matrix_mask.append([0] * (len(local_cost_matrix[i])))

    row_index = 0
    column_index = 0

    while row_index != len(local_path_matrix) - 1 and column_index != len(local_path_matrix[-1]) - 1:
        min_value = min(local_path_matrix[row_index][-1], local_path_matrix[-1][column_index])
        local_path_matrix[row_index][column_index] = min_value
        local_matrix_mask[row_index][column_index] = 1
        local_path_matrix[row_index][-1] -= min_value
        local_path_matrix[-1][column_index] -= min_value

        if local_path_matrix[row_index][-1] != local_path_matrix[-1][column_index]:
            for column_index in range(0, len(local_path_matrix[-1])):
                if local_path_matrix[-1][column_index] != 0:
                    break

        for row_index in range(0, len(local_path_matrix)):
            if local_path_matrix[row_index][-1] != 0:
                break

    sum_value = 0
    for i in range(0, len(local_cost_matrix)):
        for j in range(0, len(local_cost_matrix[0])):
            sum_value += local_path_matrix[i][j] * local_cost_matrix[i][j]
    local_path_matrix[-1][-1] = sum_value

    return local_path_matrix, local_matrix_mask


def potentials_method(local_path_matrix, local_cost_matrix, local_matrix_mask):
    local_proposal_potentials = [np.NaN] * (len(local_path_matrix) - 1)
    local_demand_potentials = [np.NaN] * (len(local_path_matrix[0]) - 1)

    local_proposal_potentials[0] = 0
    local_is_column_calculate = True
    local_last_indexes = [0]
    local_changed_nums = 1

    while local_changed_nums > 0:
        local_new_indexes = []
        local_changed_nums = 0

        if local_is_column_calculate:
            for row_index in local_last_indexes:
                for column_index in range(0, len(local_cost_matrix[0])):
                    if local_matrix_mask[row_index][column_index] != 0:
                        if np.isnan(local_demand_potentials[column_index]):
                            local_demand_potentials[column_index] = (local_cost_matrix[row_index][column_index] -
                                                                     local_proposal_potentials[row_index])
                            local_new_indexes.append(column_index)
                            local_changed_nums += 1
                            local_is_column_calculate = False
        else:
            for column_index in local_last_indexes:
                for row_index in range(0, len(local_cost_matrix)):
                    if local_matrix_mask[row_index][column_index] != 0:
                        if np.isnan(local_proposal_potentials[row_index]):
                            local_proposal_potentials[row_index] = (local_cost_matrix[row_index][column_index] -
                                                                    local_demand_potentials[column_index])
                            local_new_indexes.append(row_index)
                            local_changed_nums += 1
                            local_is_column_calculate = True
        local_last_indexes = local_new_indexes

        if local_changed_nums == 0:
            for potential in local_proposal_potentials:
                if np.isnan(potential):
                    print("Error")
                    return
            for potential in local_demand_potentials:
                if np.isnan(potential):
                    print("Error")
                    return

    local_path_deltas = []
    local_max_delta = 0
    local_max_delta_position = []
    local_is_optimal = True

    local_m = len(local_cost_matrix[0])
    for i in range(0, len(local_cost_matrix)):
        local_path_deltas.append([0] * local_m)

    for column_index in range(0, len(local_path_deltas[0])):
        for row_index in range(0, len(local_path_deltas)):
            if local_matrix_mask[row_index][column_index] == 0:
                print(local_path_deltas, local_proposal_potentials, local_demand_potentials)
                local_path_deltas[row_index][column_index] = (local_demand_potentials[column_index] +
                                                              local_proposal_potentials[row_index] -
                                                              local_cost_matrix[row_index][column_index])
                print(local_demand_potentials[column_index], local_proposal_potentials[row_index],
                      local_cost_matrix[row_index][column_index],
                      local_demand_potentials[column_index] + local_proposal_potentials[row_index] -
                      local_cost_matrix[row_index][column_index])
                if local_path_deltas[row_index][column_index] > 0:
                    if local_max_delta <= local_path_deltas[row_index][column_index]:
                        local_max_delta = local_path_deltas[row_index][column_index]
                        local_max_delta_position = [row_index, column_index]
                    local_is_optimal = False

    if local_is_optimal:
        return True, local_path_matrix, local_matrix_mask, local_proposal_potentials, local_demand_potentials, local_path_deltas, [], []

    local_paths_queue = queue.Queue()

    local_paths_queue.put(Path([], [local_max_delta_position[0], local_max_delta_position[1], 1]))
    local_paths_queue.put(Path([], [local_max_delta_position[0], local_max_delta_position[1], -1]))

    local_cycle_history = []
    local_cycle = []
    while not local_paths_queue.empty():
        path = local_paths_queue.get()
        local_return_flag, local_possible_moves = path.get_possible_moves(local_matrix_mask)

        if local_return_flag:
            local_cycle = path.way
            break

        for move in local_possible_moves:
            local_paths_queue.put(Path(path.way.copy(), move))
            for poPath in local_paths_queue.queue:
                step_queue = []
                step_queue.append(poPath.way)
            local_cycle_history.append(step_queue)

    if len(local_cycle) < 4:
        print("Error!")
        return

    local_negative_points = []
    for point_index in range(0, len(local_cycle)):
        if point_index % 2 == 0:
            local_cycle[point_index][2] = 1
        else:
            local_negative_points.append(local_path_matrix[local_cycle[point_index][0]][local_cycle[point_index][1]])
            local_cycle[point_index][2] = -1

    local_min_negative = min(local_negative_points)

    local_is_removed_duplicate = False
    for point_index in range(0, len(local_cycle)):
        if point_index % 2 == 1:
            local_path_matrix[local_cycle[point_index][0]][local_cycle[point_index][1]] += local_cycle[point_index][
                                                                                               2] * local_min_negative
            if local_path_matrix[local_cycle[point_index][0]][local_cycle[point_index][1]] == 0:
                if not local_is_removed_duplicate:
                    local_matrix_mask[local_cycle[point_index][0]][local_cycle[point_index][1]] = 0
                    local_is_removed_duplicate = True
                else:
                    local_matrix_mask[local_cycle[point_index][0]][local_cycle[point_index][1]] = 1
            else:
                local_matrix_mask[local_cycle[point_index][0]][local_cycle[point_index][1]] = 1

    for point_index in range(0, len(local_cycle)):
        if point_index % 2 == 0:
            local_path_matrix[local_cycle[point_index][0]][local_cycle[point_index][1]] += local_cycle[point_index][
                                                                                               2] * local_min_negative
            if local_path_matrix[local_cycle[point_index][0]][local_cycle[point_index][1]] == 0:
                if not local_is_removed_duplicate:
                    local_matrix_mask[local_cycle[point_index][0]][local_cycle[point_index][1]] = 0
                    local_is_removed_duplicate = True
                else:
                    local_matrix_mask[local_cycle[point_index][0]][local_cycle[point_index][1]] = 1
            else:
                local_matrix_mask[local_cycle[point_index][0]][local_cycle[point_index][1]] = 1

    local_sum_value = 0
    for i in range(0, len(local_cost_matrix)):
        for j in range(0, len(local_cost_matrix[0])):
            local_sum_value += local_path_matrix[i][j] * local_cost_matrix[i][j]
    local_path_matrix[-1][-1] = local_sum_value

    return (local_is_optimal, local_path_matrix, local_matrix_mask, local_proposal_potentials, local_demand_potentials,
            local_path_deltas, local_cycle, local_cycle_history)


###


def get_canonical(local_cost_matrix, local_proposal_vector, local_demand_vector):
    local_canonical_matrix = []
    local_basis = []
    local_matrix_less = []

    for i in range(0, len(local_demand_vector)):
        local_canonical_matrix.append([0] * (len(local_cost_matrix) * len(local_cost_matrix[0]) +
                                             len(local_proposal_vector)))
        local_matrix_less.append([0] * (len(local_cost_matrix) * len(local_cost_matrix[0]) + 1))
        for j in range(0, len(local_cost_matrix)):
            local_canonical_matrix[-1][i + (j * len(local_demand_vector))] = 1
            local_matrix_less[-1][i + (j * len(local_demand_vector))] = 1
        local_basis.append(local_demand_vector[i])
        local_matrix_less[-1][-1] = local_demand_vector[i]

    for i in range(0, len(local_proposal_vector)):
        local_canonical_matrix.append([0] * (len(local_cost_matrix) * len(local_cost_matrix[0]) +
                                             len(local_proposal_vector)))
        local_matrix_less.append([0] * (len(local_cost_matrix) * len(local_cost_matrix[0]) + 1))
        for j in range(0, len(local_cost_matrix[i])):
            local_canonical_matrix[-1][j + (i * len(local_demand_vector))] = 1
            local_matrix_less[-1][j + (i * len(local_demand_vector))] = 1
        local_canonical_matrix[-1][len(local_cost_matrix) * len(local_cost_matrix[0]) + i] = 1
        local_basis.append(local_proposal_vector[i])
        local_matrix_less[-1][-1] = local_proposal_vector[i]

    local_function_vector = [0] * (len(local_cost_matrix) * len(local_cost_matrix[0]) + len(local_proposal_vector))
    for i in range(0, len(local_proposal_vector)):
        for j in range(0, len(local_cost_matrix[i])):
            local_function_vector[j + (i * len(local_cost_matrix[i]))] = -1 * local_cost_matrix[i][j]

    local_x_limits = [1] * (len(local_cost_matrix) * len(local_cost_matrix[0]) + len(local_proposal_vector))

    return local_canonical_matrix, local_basis, local_function_vector, local_x_limits, [], local_matrix_less


def bubble_max_row(m, col):
    max_element = m[col][col]
    max_row = col
    for i in range(col + 1, len(m)):
        if abs(m[i][col]) > abs(max_element):
            max_element = m[i][col]
            max_row = i
    if max_row != col:
        temp_row = m[col].copy()
        m[col] = m[max_row]
        m[max_row] = temp_row


def solve_gauss(m):
    n = len(m)

    for k in range(n - 1):
        bubble_max_row(m, k)
        for i in range(k + 1, n):
            div = m[i][k] / m[k][k]
            m[i][-1] -= div * m[k][-1]
            for j in range(k, n):
                m[i][j] -= div * m[k][j]

    if is_singular(m):
        return

    x = [0 for i in range(n)]
    for k in range(n - 1, -1, -1):
        x[k] = (m[k][-1] - sum([m[k][j] * x[j] for j in range(k + 1, n)])) / m[k][k]

    return x


def is_singular(m):
    for i in range(len(m)):
        if not m[i][i]:
            return True
    return False


def brute_force(local_matrix, local_function, local_basis, local_x_limits, local_matrix_equality, local_matrix_more,
                local_matrix_less):
    n = len(local_x_limits)
    answer_result = -math.inf
    answer_vector = []
    saved_column_indices = []

    local_x_limits += [1] * (len(local_function) - len(local_x_limits))

    column_indices = []
    for i in range(0, len(local_function)):
        column_indices.append(i)

    comb_generator = itertools.combinations(column_indices, len(local_matrix))

    iterations = 0

    try:
        while True:
            used_columns = next(comb_generator)
            sle = []
            for i in range(0, len(local_matrix)):
                row = []
                for j in range(0, len(used_columns)):
                    row.append(local_matrix[i][used_columns[j]])
                sle.append(row)
            dot = np.linalg.det(sle)
            iterations += 1
            if dot != 0:
                for i in range(0, len(sle)):
                    sle[i].append(local_basis[i])
                try:
                    x = solve_gauss(sle.copy())
                    limits_flag = True

                    for k in range(0, len(used_columns)):
                        if local_x_limits[used_columns[k]] == 1:
                            if x[k] < 0:
                                print("x!")
                                limits_flag = False
                        elif local_x_limits[used_columns[k]] == -1:
                            if x[k] > 0:
                                limits_flag = False

                    for k in range(0, len(local_matrix_equality)):
                        temp_result = 0
                        for j in range(0, len(used_columns)):
                            temp_result += (local_matrix[k][used_columns[j]] * x[j])
                        if abs(temp_result - local_basis[k]) > 0.0000001:
                            print("equality!")
                            limits_flag = False

                    for k in range(0, len(local_matrix_less)):
                        temp_result = 0
                        for j in range(0, len(used_columns)):
                            temp_result += (local_matrix[len(local_matrix_equality) + k][used_columns[j]] * x[j])
                        if temp_result > local_basis[len(local_matrix_equality) + k]:
                            print("less!")
                            limits_flag = False

                    for k in range(0, len(local_matrix_more)):
                        temp_result = 0
                        for j in range(0, len(used_columns)):
                            temp_result += (
                                    local_matrix[len(local_matrix_equality) + len(local_matrix_less) + k][
                                        used_columns[j]] * x[j])
                        if temp_result < local_basis[len(local_matrix_equality) + len(local_matrix_less) + k]:
                            print("more!")
                            limits_flag = False

                    if limits_flag:
                        reference_vector = [0] * len(local_function)
                        for i in range(0, len(used_columns)):
                            reference_vector[used_columns[i]] = x[i]
                        temp_result = 0
                        for k in range(0, len(used_columns)):
                            temp_result += (local_function[used_columns[k]] * x[k])
                        if answer_result < temp_result:
                            answer_result = temp_result
                            answer_vector = x.copy()
                            saved_column_indices = used_columns
                        print(reference_vector, temp_result)
                except:
                    continue
    except StopIteration:
        reference_vector = [0] * len(local_function)
        for i in range(0, len(saved_column_indices)):
            reference_vector[saved_column_indices[i]] = answer_vector[i]
        j = 0
        result = [0] * n
        for i in range(0, n):
            if local_x_limits[i] == 0:
                result[i] = reference_vector[i] - reference_vector[n + j]
                j += 1
            else:
                result[i] = reference_vector[i]
        print(result, iterations)
        return result, iterations


def getDouble(local_matrix, local_function, local_basis):
    A_ub_T = np.array(local_matrix).transpose()
    return A_ub_T, local_basis, local_function


###


if __name__ == '__main__':
    advantage = sum_balance_check(proposal_vector, demand_vector)
    if advantage < 0:
        proposal_vector.append(abs(advantage))
        cost_matrix.append([0] * len(cost_matrix[0]))
    elif advantage > 0:
        demand_vector.append(abs(advantage))
        for i in range(0, len(cost_matrix)):
            cost_matrix[i].append(0)

    is_optimal = False
    path_matrix, matrix_mask = northwest_method(cost_matrix, proposal_vector, demand_vector)

    while not is_optimal:
        (is_optimal, path_matrix, matrix_mask, proposal_potentials, demand_potentials, path_deltas,
         cycle) = potentials_method(path_matrix, cost_matrix, matrix_mask)
        print(np.array(path_matrix))
        print(np.array(matrix_mask))
        print(is_optimal)

    canonical_matrix, basis, function_vector, x_limits, matrix_equality, matrix_less = get_canonical(
        cost_matrix, proposal_vector, demand_vector)
    print(brute_force(canonical_matrix, function_vector, basis, x_limits, matrix_equality, [], matrix_less))
