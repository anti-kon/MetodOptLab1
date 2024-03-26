import copy
import itertools
import math
import numpy as np


# def get_standard_form(local_matrix_equality, local_matrix_more, local_matrix_less, local_x_limits,
#                       local_target_func, local_is_min):
#     if local_is_min:
#         local_function = [element * -1 for element in local_target_func]
#     else:
#         local_function = copy.deepcopy(local_target_func)
#
#
#     local_matrix = []
#     local_basis = []
#     for equality in local_matrix_equality:
#         local_matrix_more.append(equality)
#         local_matrix_less.append(equality)
#     for row_index in range(0, len(local_matrix_more)):
#         local_matrix.append([0] * (len(local_matrix_more[row_index]) - 1))
#         local_basis.append((1 if local_is_min else -1) * local_matrix_more[row_index][-1])
#         for column_index in range(0, len(local_matrix_more[row_index]) - 1):
#             local_matrix[-1][column_index] = (1 if local_is_min else -1) * local_matrix_more[row_index][column_index]
#     for row_index in range(0, len(local_matrix_less)):
#         local_matrix.append([0] * (len(local_matrix_less[row_index]) - 1))
#         local_basis.append((-1 if local_is_min else 1) * local_matrix_less[row_index][-1])
#         for column_index in range(0, len(local_matrix_less[row_index]) - 1):
#             local_matrix[-1][column_index] = (-1 if local_is_min else 1) * local_matrix_less[row_index][column_index]
#
#     for x_index in range(len(local_x_limits)):
#         if local_x_limits[x_index] < 0:
#             for row in range(len(local_matrix)):
#                 local_matrix[row][x_index] *= -1
#             local_function[x_index] *= -1
#         if local_x_limits[x_index] == 0:
#             for row in range(len(local_matrix)):
#                 local_matrix[row].append(-1 * local_matrix[row][x_index])
#             local_function.append(local_function[x_index] * -1)
#
#     return local_matrix, local_basis, local_function
#
#
# def get_canonical_form(local_matrix, local_basis, local_function, local_is_min):
#     local_used_columns = []
#     for row_index in range(0, len(local_matrix)):
#         local_function.append(0)
#         for column_index in range(0, len(local_matrix)):
#             if row_index == column_index:
#                 local_matrix[row_index].append(-1 if local_is_min else 1)
#                 local_used_columns.append(len(local_matrix) - 1 + column_index)
#             else:
#                 local_matrix[row_index].append(0)
#         if local_basis[row_index] < 0:
#             for column_index in range(0, len(local_matrix[row_index])):
#                 local_matrix[row_index][column_index] *= -1
#             local_basis[row_index] *= -1
#
#     return local_matrix, local_basis, local_function, local_used_columns


def get_canonical_form(local_matrix_equality, local_matrix_more, local_matrix_less, local_x_limits,
                       local_target_func, local_is_min):
    if local_is_min:
        local_function = [element * -1 for element in local_target_func]
    else:
        local_function = copy.deepcopy(local_target_func)

    local_matrix = []
    local_basis = []
    for row_index in range(0, len(local_matrix_equality)):
        local_matrix.append([0] * (len(local_matrix_equality[row_index]) - 1))
        local_basis.append(local_matrix_equality[row_index][-1])
        for column_index in range(0, len(local_matrix_equality[row_index]) - 1):
            local_matrix[-1][column_index] = local_matrix_equality[row_index][column_index]
    for row_index in range(0, len(local_matrix_more)):
        local_matrix.append([0] * (len(local_matrix_more[row_index]) - 1))
        local_basis.append(local_matrix_more[row_index][-1])
        for column_index in range(0, len(local_matrix_more[row_index]) - 1):
            local_matrix[-1][column_index] = local_matrix_more[row_index][column_index]
    for row_index in range(0, len(local_matrix_less)):
        local_matrix.append([0] * (len(local_matrix_less[row_index]) - 1))
        local_basis.append(local_matrix_less[row_index][-1])
        for column_index in range(0, len(local_matrix_less[row_index]) - 1):
            local_matrix[-1][column_index] = local_matrix_less[row_index][column_index]

    for x_index in range(len(local_x_limits)):
        if local_x_limits[x_index] < 0:
            for row in range(len(local_matrix)):
                local_matrix[row][x_index] *= -1
                if row < len(local_matrix_equality):
                    local_matrix_equality[row][x_index] *= -1
                if row < len(local_matrix_more):
                    local_matrix_more[row][x_index] *= -1
                if row < len(local_matrix_less):
                    local_matrix_less[row][x_index] *= -1
            local_function[x_index] *= -1
        if local_x_limits[x_index] == 0:
            for row in range(len(local_matrix)):
                local_matrix[row].append(-1 * local_matrix[row][x_index])
            local_function.append(local_function[x_index] * -1)

    for row_index in range(0, len(local_matrix_more)):
        local_function.append(0)
        for column_index in range(0, len(local_matrix)):
            if row_index == column_index:
                local_matrix[row_index].append(-1)
            else:
                local_matrix[row_index].append(0)
    for row_index in range(len(local_matrix_more), len(local_matrix_more) + len(local_matrix_less)):
        local_function.append(0)
        for column_index in range(0, len(local_matrix)):
            if row_index == column_index:
                local_matrix[row_index].append(1)
            else:
                local_matrix[row_index].append(0)
    for row_index in range(len(local_matrix_more) + len(local_matrix_less),
                           len(local_matrix_more) + len(local_matrix_less) + len(local_matrix_equality)):
        local_function.append(0)
        for column_index in range(0, len(local_matrix)):
            local_matrix[row_index].append(0)

    return local_matrix, local_basis, local_function, local_matrix_equality, local_matrix_more, local_matrix_less


def get_basis_form(local_matrix, local_basis, local_function):
    use_additional_basis = False
    additional_rows = []
    local_used_columns = [0] * len(local_basis)
    for row_index in range(0, len(local_basis)):
        if local_basis[row_index] < 0:
            local_matrix[row_index] = [element * -1 for element in local_matrix[row_index]]
            local_basis[row_index] *= -1
    for row_index in range(0, len(local_basis)):
        have_basis = False
        save_column_index = 0
        for column_index in range(0, len(local_matrix[row_index])):
            potential_basis = True
            if local_matrix[row_index][column_index] > 0:
                for check_row_index in range(0, len(local_basis)):
                    if check_row_index != row_index and local_matrix[check_row_index][column_index] != 0:
                        potential_basis = False
                        break
            else:
                potential_basis = False
            if potential_basis:
                save_column_index = column_index
                have_basis = True
                break
        if have_basis:
            if local_matrix[row_index][column_index] != -1:
                for column_index in range(0, len(local_matrix[row_index])):
                    if column_index != save_column_index:
                        local_matrix[row_index][column_index] /= local_matrix[row_index][save_column_index]
                local_basis[row_index] /= local_matrix[row_index][save_column_index]
                local_matrix[row_index][save_column_index] = 1
            local_used_columns[row_index] = save_column_index
        else:
            use_additional_basis = True
            additional_rows.append(row_index)
            local_used_columns[row_index] = len(local_function)
            local_function.append(0)
            for new_basis_vector_row_index in range(0, len(local_basis)):
                if new_basis_vector_row_index != row_index:
                    local_matrix[new_basis_vector_row_index].append(0)
                else:
                    local_matrix[new_basis_vector_row_index].append(1)
    if not use_additional_basis:
        return local_matrix, local_basis, local_function, local_used_columns
    local_extend_matrix = []
    for row_index in additional_rows:
        local_extend_matrix.append([0] * (local_used_columns[additional_rows[0]] + 1))
        for column_index in range(0, len(local_extend_matrix[-1]) - 1):
            local_extend_matrix[-1][column_index] = local_matrix[row_index][column_index]
        local_extend_matrix[-1][-1] = local_basis[row_index]
    local_extend_matrix.append([0] * (local_used_columns[additional_rows[0]] + 1))
    for column_index in range(0, len(local_extend_matrix[-1]) - 1):
        local_extend_matrix[-1][column_index] = -1 * local_function[column_index]
    M = [0] * (local_used_columns[additional_rows[0]] + 1)
    for row_index in additional_rows:
        for column_index in range(0, len(local_extend_matrix[-1]) - 1):
            M[column_index] += local_matrix[row_index][column_index]
        M[-1] -= local_basis[row_index]
    local_extend_matrix.append(M)

    while not check_negative(local_extend_matrix[-1]):
        min_column = -1

        for i in range(0, len(local_extend_matrix[-1]) - 1):
            if is_min:
                if local_extend_matrix[-1][i] < 0 and (
                        min_column == -1 or local_extend_matrix[-1][i] < local_extend_matrix[-1][min_column]):
                    min_column = i
            else:
                if local_extend_matrix[-1][i] > 0 and (
                        min_column == -1 or local_extend_matrix[-1][i] < local_extend_matrix[-1][min_column]):
                    min_column = i

        special_coff = [-1] * (len(local_extend_matrix) - 1)

        for i in range(0, len(local_extend_matrix) - 1):
            if local_extend_matrix[i][min_column] > 0:
                special_coff[i] = local_extend_matrix[i][-1] / local_extend_matrix[i][min_column]

        min_row = -1
        for i in range(0, len(special_coff)):
            if special_coff[i] > 0:
                if min_row == -1 or special_coff[min_row] > special_coff[i]:
                    min_row = i

        if min_row == -1:
            print("None")
            return None

        for m in range(0, len(local_extend_matrix[min_row])):
            if m != min_column:
                local_extend_matrix[min_row][m] /= local_extend_matrix[min_row][min_column]
        local_extend_matrix[min_row][min_column] = 1

        for k in range(len(local_extend_matrix)):
            if k != min_row:
                div = local_extend_matrix[k][min_column] / local_extend_matrix[min_row][min_column]
                for m in range(0, len(local_extend_matrix[k])):
                    local_extend_matrix[k][m] -= div * local_extend_matrix[min_row][m]

        local_used_columns[min_row] = min_column

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


def brute_force(local_matrix, local_basis, local_function, local_x_limits, local_matrix_equality, local_matrix_more,
                local_matrix_less):
    n = len(local_x_limits)
    answer_result = -math.inf
    answer_vector = []
    saved_column_indices = []

    local_x_limits = [abs(element) for element in local_x_limits]
    local_x_limits += [1] * (len(local_function) - len(local_x_limits))

    column_indices = []
    for i in range(0, len(local_function)):
        column_indices.append(i)

    comb_generator = itertools.combinations(column_indices, len(local_matrix))

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
            if dot != 0:
                for i in range(0, len(sle)):
                    sle[i].append(local_basis[i])
                try:
                    x = solve_gauss(sle.copy())
                    limits_flag = True

                    temp_reference_vector = [0] * len(local_function)
                    for i in range(0, len(used_columns)):
                        temp_reference_vector[used_columns[i]] = x[i]
                    j = 0
                    temp_result_vector = [0] * n
                    for i in range(0, n):
                        if local_x_limits[i] == 0:
                            temp_result_vector[i] = temp_reference_vector[i] - temp_reference_vector[n + j]
                            j += 1
                        else:
                            temp_result_vector[i] = temp_reference_vector[i]

                    for k in range(0, len(used_columns)):
                        if local_x_limits[used_columns[k]] == 1:
                            if x[k] < 0:
                                limits_flag = False
                        elif local_x_limits[used_columns[k]] == -1:
                            if x[k] > 0:
                                limits_flag = False

                    for k in range(0, len(local_matrix)):
                        temp_result = 0
                        for j in range(0, len(used_columns)):
                            temp_result += (local_matrix[k][used_columns[j]] * x[j])
                        if abs(temp_result - local_basis[k]) > 0.0000001:
                            limits_flag = False

                    for k in range(0, len(local_matrix_equality)):
                        temp_result = 0
                        for j in range(0, len(temp_result_vector)):
                            temp_result += (local_matrix_equality[k][j] * temp_result_vector[j])
                        if abs(temp_result - local_matrix_equality[k][-1]) > 0.0000001:
                            limits_flag = False

                    for k in range(0, len(local_matrix_less)):
                        temp_result = 0
                        for j in range(0, len(temp_result_vector)):
                            temp_result += (local_matrix_less[k][j] * temp_result_vector[j])
                        if temp_result > float(local_matrix_less[k][-1]):
                            if limits_flag:
                                print(x, ">")
                            limits_flag = False

                    for k in range(0, len(local_matrix_more)):
                        temp_result = 0
                        for j in range(0, len(temp_result_vector)):
                            temp_result += (local_matrix_more[k][j] * temp_result_vector[j])
                        if temp_result < float(local_matrix_more[k][-1]):
                            if limits_flag:
                                print(x, "<")
                            limits_flag = False

                    if limits_flag:
                        temp_result = 0
                        for k in range(0, len(used_columns)):
                            temp_result += (local_function[used_columns[k]] * x[k])
                        if answer_result < temp_result:
                            answer_result = temp_result
                            answer_vector = x.copy()
                            saved_column_indices = used_columns
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
        return result


def simplex_method(local_matrix, local_basis, local_function, local_x_limits, local_used_columns, local_is_min):
    if local_is_min:
        for i in range(len(local_function)):
            local_function[i] *= -1

    start_x_limit_size = len(local_x_limits)
    extended_matrix = local_matrix

    for i in range(len(extended_matrix)):
        extended_matrix[i].append(local_basis[i])

    extended_matrix.append([0] * len(local_function))
    for i in range(0, len(local_matrix[-1])):
        local_matrix[-1][i] = local_function[i]
    extended_matrix[-1].append(0)

    first_try = True
    while first_try or ((not local_is_min and not check_negative(extended_matrix[-1])) or
           (local_is_min and not check_positive(extended_matrix[-1]))):
        min_column = -1
        first_try = False

        for i in range(0, len(extended_matrix[-1]) - 1):
            if is_min:
                if extended_matrix[-1][i] < 0 and (min_column == -1 or extended_matrix[-1][i] < extended_matrix[-1][min_column]):
                    min_column = i
            else:
                if extended_matrix[-1][i] > 0 and (min_column == -1 or extended_matrix[-1][i] < extended_matrix[-1][min_column]):
                    min_column = i

        special_coff = [-1] * (len(extended_matrix) - 1)

        for i in range(0, len(extended_matrix) - 1):
            if extended_matrix[i][min_column] > 0:
                special_coff[i] = extended_matrix[i][-1] / extended_matrix[i][min_column]

        min_row = -1
        for i in range(0, len(special_coff)):
            if special_coff[i] > 0:
                if min_row == -1 or special_coff[min_row] > special_coff[i]:
                    min_row = i

        if min_row == -1:
            print("None")
            return None

        for m in range(0, len(extended_matrix[min_row])):
            if m != min_column:
                extended_matrix[min_row][m] /= extended_matrix[min_row][min_column]
        extended_matrix[min_row][min_column] = 1

        for k in range(len(extended_matrix)):
            if k != min_row:
                div = extended_matrix[k][min_column] / extended_matrix[min_row][min_column]
                for m in range(0, len(extended_matrix[k])):
                    extended_matrix[k][m] -= div * extended_matrix[min_row][m]

        local_used_columns[min_row] = min_column

    reference_vector = [0] * len(local_function)
    for i in range(0, len(extended_matrix) - 1):
        reference_vector[local_used_columns[i]] = extended_matrix[i][-1]
    j = 0
    result = [0] * start_x_limit_size
    for i in range(0, start_x_limit_size):
        if local_x_limits[i] == 0:
            result[i] = reference_vector[i] - reference_vector[len(local_x_limits) + j]
            j += 1
        else:
            result[i] = reference_vector[i]
    return result


def check_positive(vector):
    answer = True
    for i in range(0, len(vector)):
        if (vector[i] < 0):
            answer = False
    return answer

def check_negative(vector):
    answer = True
    for i in range(0, len(vector)):
        if (vector[i] > 0):
            answer = False
    return answer


def get_double_task(local_matrix_equality, local_matrix_more, local_matrix_less, local_x_limits, local_target_func,
                    local_is_min):
    local_double_x_limits = [0] * (len(local_matrix_equality) + len(local_matrix_more) + len(local_matrix_less))
    for index_relationship_more in range(len(local_matrix_equality),
                                         len(local_matrix_equality) + len(local_matrix_more)):
        local_double_x_limits[index_relationship_more] = -1 if not local_is_min else 1
    for index_relationship_less in range(len(local_matrix_equality) + len(local_matrix_more),
                                         len(local_matrix_equality) + len(local_matrix_more) + len(local_matrix_less)):
        local_double_x_limits[index_relationship_less] = 1 if not local_is_min else -1

    local_is_double_min = not local_is_min
    local_double_function = [0] * (len(local_matrix_equality) + len(local_matrix_more) + len(local_matrix_less))
    for row_index in range(0, len(local_matrix_equality)):
        local_double_function[row_index] = local_matrix_equality[row_index][-1]
    for row_index in range(0, len(local_matrix_less)):
        local_double_function[len(local_matrix_equality) + row_index] = local_matrix_less[row_index][-1]
    for row_index in range(0, len(local_matrix_more)):
        local_double_function[len(local_matrix_equality) + len(local_matrix_less)
                        + row_index] = local_matrix_more[row_index][-1]

    matrix = []
    for row in range(0, len(local_matrix_equality)):
        matrix.append([0] * (len(local_matrix_equality[row]) - 1))
        for column in range(0, len(local_matrix_equality[row]) - 1):
            matrix[row][column] = local_matrix_equality[row][column]
    for row in range(0, len(local_matrix_less)):
        matrix.append([0] * (len(local_matrix_less[row]) - 1))
        for column in range(0, len(local_matrix_less[row]) - 1):
            matrix[row][column] = local_matrix_less[row][column]
    for row in range(0, len(local_matrix_more)):
        matrix.append([0] * (len(local_matrix_more[row]) - 1))
        for column in range(0, len(local_matrix_more[row]) - 1):
            matrix[row][column] = local_matrix_more[row][column]

    matrix = np.array(matrix).transpose()

    local_double_matrix_equality = []
    local_double_matrix_more = []
    local_double_matrix_less = []

    for x_limit_index in range(0, len(local_x_limits)):
        if local_x_limits[x_limit_index] == -1:
            if not local_is_min:
                local_double_matrix_less.append([0] * len(matrix[x_limit_index]))
                for column_index in range(0, len(matrix[x_limit_index])):
                        local_double_matrix_less[-1][column_index] = matrix[x_limit_index][column_index]
                local_double_matrix_less[-1].append(local_target_func[x_limit_index])
            else:
                local_double_matrix_more.append([0] * len(matrix[x_limit_index]))
                for column_index in range(0, len(matrix[x_limit_index])):
                    local_double_matrix_more[-1][column_index] = matrix[x_limit_index][column_index]
                local_double_matrix_more[-1].append(local_target_func[x_limit_index])
        elif local_x_limits[x_limit_index] == 1:
            if not local_is_min:
                local_double_matrix_more.append([0] * len(matrix[x_limit_index]))
                for column_index in range(0, len(matrix[x_limit_index])):
                    local_double_matrix_more[-1][column_index] = matrix[x_limit_index][column_index]
                local_double_matrix_more[-1].append(local_target_func[x_limit_index])
            else:
                local_double_matrix_less.append([0] * len(matrix[x_limit_index]))
                for column_index in range(0, len(matrix[x_limit_index])):
                        local_double_matrix_less[-1][column_index] = matrix[x_limit_index][column_index]
                local_double_matrix_less[-1].append(local_target_func[x_limit_index])
        else:
            local_double_matrix_equality.append([0] * len(matrix[x_limit_index]))
            for column_index in range(0, len(matrix[x_limit_index])):
                local_double_matrix_equality[-1][column_index] = matrix[x_limit_index][column_index]
            local_double_matrix_equality[-1].append(local_target_func[x_limit_index])
    return (local_double_matrix_equality, local_double_matrix_more, local_double_matrix_less, local_double_x_limits,
            local_double_function, local_is_double_min)


# # ==
# A = [[-8, 3, 1, -2, -2, -2, 23], [-4, 2, 1, -1, -1, -1, 20], [-4, 2, 2, -1, -1, -1, 29]]
#
# # >=
# B = [[13, -3, -1, 3, 3, 3, -13]]
#
# # <=
# C = [[-23, 3, 1, -2, -3, -5, 17], [-18, 3, 1, -3, -3, -4, 12]]
#
# F = [4, 6, 2, 6, 5, 3]

# # ==
# A = [[1, -1, -4, 2, 1, 5], [1, 1, 1, 3, 2, 9], [1, 1, 1, 2, 1, 6]]
#
# # >=
# B = [[5, 9, -10, 4, 20, -4], [1, 1, -14, -7, 9, 25]]
#
# # <=
# C = [[2, -9, -7, 10, -1, 2]]
#
# F = [1, 2, -1, 3, -1]

# # ==
# A = []
#
# B = []
#
# C = [[6, 3, 1, 4, 252], [2, 4, 5, 1, 144], [1, 2, 4, 3, 80]]
#
# F = [48, 33, 16, 22]
#
# # 0 - not, 1 -> xi > 0, -1 -> xi < 0
# xLimits = [1, 1, 1, 1]

#-------------------------------------------
#
# # ==
# A = []
#
# B = []
#
# # <=
# C = [[1, 1, 60], [2, 1, 80], [0, 1, 20]]
#
# F = [-1, -1]
#
# # 0 - not, 1 -> xi > 0, -1 -> xi < 0
# xLimits = [1, 1]
#
# is_min = True

# # ==
# A = []
#
# B = []
#
# # <=
# C = [[1, 1, 55], [2, 3, 120], [12, 30, 960]]
#
# F = [3, 4]
#
# # 0 - not, 1 -> xi > 0, -1 -> xi < 0
# xLimits = [1, 1]
#
# is_min = False

# # ==
# A = []
#
# B = []
#
# # <=
# C = [[1, 2, 6], [2, 1, 8], [0, 1, 2]]
#
# F = [1, 2]
#
# # 0 - not, 1 -> xi > 0, -1 -> xi < 0
# xLimits = [1, 1]
#
# is_min = False

# # ==
# A = []
#
# B = []
#
# # <=
# C = [[-1, 2, 1], [1, -2, 2]]
#
# F = [1, 1]
#
# # 0 - not, 1 -> xi > 0, -1 -> xi < 0
# xLimits = [1, 1]
#
# is_min = False

# # ==
# A = []
#
# B = []
#
# # <=
# C = [[-1, 2, 1], [1, -2, 2]]
#
# F = [-1, -2]
#
# # 0 - not, 1 -> xi > 0, -1 -> xi < 0
# xLimits = [1, 1]
#
# is_min = True


# # ==
# A = []
#
# B = [[1, -3, 6]]
#
# # <=
# C = [[3, 5, 30], [4, -3, 12]]
#
# F = [1, 1]
#
# # 0 - not, 1 -> xi > 0, -1 -> xi < 0
# xLimits = [1, 1]
#
#is_min = False

# # ==
# A = []
#
# B = [[1, 2, 12, 3], [1, 3, 30, 4]]
#
# # <=
# C = []
#
# F = [55, 120, 960]
#
# # 0 - not, 1 -> xi > 0, -1 -> xi < 0
# xLimits = [1, 1, 1]
#
# is_min = True

# # ==
# A = [[2, 1, 4, 7, 16, 12], [3, 3, 5, 11, 24, 20], [4, 5, 7, 16, 35, 29]]
#
# B = [[-5, 9, 8, 1, -30, -531], [-2, 6, -7, 2, -25, -286]]
#
# # <=
# C = [[-9, 2, 3, -3, -24, 892]]
#
# F = [1, 1, 1, 1, 1]
#
# # 0 - not, 1 -> xi > 0, -1 -> xi < 0
# xLimits = [0, 0, 0, 0, 0]
#
# is_min = False


# ==
A = [[-8, 3, 1, -2, -2, -2, 23], [-4, 2, 1, -1, -1, -1, 20], [-4, 2, 2, -1, -1, -1, 29]]

B = [[13, -3, -1, 3, 3, 3, -13]]

# <=
C = [[-23, 3, 1, -2, -3, -5, 17], [-18, 3, 1, -3, -3, -4, 12]]

F = [4, 6, 2, 6, 5, 3]

# 0 - not, 1 -> xi > 0, -1 -> xi < 0
xLimits = [1, 1, 1, 0, 0]

is_min = True


if __name__ == '__main__':
    matrix, basis, function, matrix_equality, matrix_more, matrix_less = get_canonical_form(
        copy.deepcopy(A), copy.deepcopy(B), copy.deepcopy(C), copy.deepcopy(xLimits), copy.deepcopy(F), is_min)
    matrix, basis, function, used_columns = get_basis_form(matrix, basis, function)
    print(np.array(matrix), basis, function)
    print(simplex_method(matrix, basis, function, copy.deepcopy(xLimits), used_columns, is_min))
    matrix, basis, function, matrix_equality, matrix_more, matrix_less = get_canonical_form(
        copy.deepcopy(A), copy.deepcopy(B), copy.deepcopy(C), copy.deepcopy(xLimits), copy.deepcopy(F), is_min)
    matrix, basis, function, used_columns = get_basis_form(matrix, basis, function)
    print(brute_force(matrix, basis, function, copy.deepcopy(xLimits), matrix_equality, matrix_more, matrix_less))

    double_matrix_equality, double_matrix_more, double_matrix_less, double_x_limits, double_function, is_double_min = (
        get_double_task(copy.deepcopy(A), copy.deepcopy(B), copy.deepcopy(C),
                        copy.deepcopy(xLimits), copy.deepcopy(F), is_min))
    print(double_matrix_equality, double_matrix_more, double_matrix_less, double_x_limits, double_function,
          is_double_min)
    matrix, basis, function, matrix_equality, matrix_more, matrix_less = get_canonical_form(
        copy.deepcopy(double_matrix_equality), copy.deepcopy(double_matrix_more), copy.deepcopy(double_matrix_less),
        copy.deepcopy(double_x_limits), copy.deepcopy(double_function), is_double_min)
    print(np.array(matrix), basis, function)
    matrix, basis, function, used_columns = get_basis_form(matrix, basis, function)
    print(np.array(matrix), basis, function)
    print(simplex_method(matrix, basis, function, copy.deepcopy(double_x_limits), used_columns, is_double_min))
    matrix, basis, function, matrix_equality, matrix_more, matrix_less = get_canonical_form(
        copy.deepcopy(double_matrix_equality), copy.deepcopy(double_matrix_more), copy.deepcopy(double_matrix_less),
        copy.deepcopy(double_x_limits), copy.deepcopy(double_function), is_double_min)
    matrix, basis, function, used_columns = get_basis_form(matrix, basis, function)
    print("!")
    print(brute_force(matrix, basis, function, copy.deepcopy(double_x_limits), matrix_equality, matrix_more,
                      matrix_less))
    #
    # i11 = [3]
    # i12 = [2]
    # i13 = [-2]
    # i14 = [5]
    # i15 = [1]
    # i21 = range(-10, 11)
    # i22 = range(-10, 11)
    # i23 = range(-10, 11)
    # i24 = range(-10, 11)
    # i25 = range(-10, 11)
    # i31 = range(-10, 11)
    # i32 = range(-10, 11)
    # i33 = range(-10, 11)
    # i34 = range(-10, 11)
    # i35 = range(-10, 11)
    # i41 = [2]
    # i42 = [-1]
    # i43 = [1]
    # i44 = [2]
    # i45 = range(-10, 11)
    # i51 = [-1]
    # i52 = [3]
    # i53 = [5]
    # i54 = [1]
    # i55 = range(-10, 11)
    # i61 = range(-10, 11)
    # i62 = range(-10, 11)
    # i63 = range(-10, 11)
    # i64 = range(-10, 11)
    # i65 = range(-10, 11)
    # b1 = [30]
    # b2 = range(-10, 11)
    # b3 = range(-10, 11)
    # b4 = [12]
    # b5 = [16]
    # b6 = range(-10, 11)
    # c1 = [5]
    # c2 = [-2]
    # c3 = [-6]
    # c4 = [4]
    # c5 = [2]
    # for j1 in range(0, 5):
    #     for j2 in range(j1 + 1, 5):
    #         for j3 in range(j2 + 1, 5):
    #             for js1 in range(-1, 1):
    #                 for js2 in range(-1, 1):
    #                     for js3 in range(-1, 1):
    #                         task_x_limits = [0] * 5
    #                         task_x_limits[j1] = js1 if js1 < 0 else 1
    #                         task_x_limits[j2] = js2 if js2 < 0 else 1
    #                         task_x_limits[j3] = js3 if js3 < 0 else 1
    #                         for x in (itertools.product(i11, i12, i13, i14, i15, b1,
    #                                                     i21, i22, i23, i24, i25, b2,
    #                                                     i31, i32, i33, i34, i35, b3,
    #                                                     i41, i42, i43, i44, i45, b4,
    #                                                     i51, i52, i53, i54, i55, b5,
    #                                                     i61, i62, i63, i64, i65, b6,
    #                                                     c1, c2, c3, c4, c5)):
    #                             for index in range(0, 41):
    #                                 if x[index] == 0:
    #                                     break
    #                             task_A = [[x[0], x[1], x[2], x[3], x[4], x[5]],
    #                                       [x[6], x[7], x[8], x[9], x[10], x[11]],
    #                                       [x[12], x[13], x[14], x[15], x[16], x[17]]]
    #                             task_B = [[x[18], x[19], x[20], x[21], x[22], x[23]],
    #                                       [x[24], x[25], x[26], x[27], x[28], x[29]]]
    #                             task_C = [[x[30], x[31], x[32], x[33], x[34], x[35]]]
    #                             task_F = [x[36], x[37], x[38], x[39], x[40]]
    #                             task_is_min = False
    #                             matrix, basis, function, matrix_equality, matrix_more, matrix_less = get_canonical_form(
    #                                 copy.deepcopy(task_A), copy.deepcopy(task_B), copy.deepcopy(task_C),
    #                                 copy.deepcopy(task_x_limits), copy.deepcopy(task_F), task_is_min)
    #                             matrix, basis, function, used_columns = get_basis_form(matrix, basis, function)
    #                             br = brute_force(matrix, basis, function, copy.deepcopy(task_x_limits),
    #                                              matrix_equality, matrix_more, matrix_less)
    #                             matrix, basis, function, matrix_equality, matrix_more, matrix_less = get_canonical_form(
    #                                 copy.deepcopy(task_A), copy.deepcopy(task_B), copy.deepcopy(task_C),
    #                                 copy.deepcopy(task_x_limits), copy.deepcopy(task_F), task_is_min)
    #                             matrix, basis, function, used_columns = get_basis_form(matrix, basis, function)
    #                             sm = simplex_method(matrix, basis, function, copy.deepcopy(task_x_limits),
    #                                                 used_columns, task_is_min)
    #                             if sm is not None and functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, br, sm), True):
    #                                 print("!")
    #                                 print(sm)
    #                                 print(np.array(task_A))
    #                                 print(np.array(task_B))
    #                                 print(np.array(task_C))
    #                                 print(np.array(task_F))
    #                                 print(task_x_limits)
    #                                 print(task_is_min)
    #                                 exit(4)
    #
