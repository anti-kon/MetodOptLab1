import itertools
import math
import numpy as np

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

def brute_force(matrix, function, basis, x_limits, matrix_equality, matrix_more, matrix_less):
    n = len(x_limits)
    answer_result = -math.inf
    answer_vector = []
    saved_column_indices = []

    x_limits += [0] * (len(function) - len(x_limits))

    column_indices = []
    for i in range(0, len(function)):
        column_indices.append(i)

    comb_generator = itertools.combinations(column_indices, len(matrix))

    try:
        while True:
            used_columns = next(comb_generator)
            sle = []
            for i in range(0, len(matrix)):
                row = []
                for j in range(0, len(used_columns)):
                    row.append(matrix[i][used_columns[j]])
                sle.append(row)
            dot = np.linalg.det(sle)
            if dot != 0:
                for i in range(0, len(sle)):
                    sle[i].append(basis[i])
                try:
                    x = solve_gauss(sle.copy())
                    limits_flag = True

                    for k in range(0, len(used_columns)):
                        if x_limits[used_columns[k]] == 1:
                            if x[k] < 0:
                                limits_flag = False
                        elif x_limits[used_columns[k]] == -1:
                            if x[k] > 0:
                                limits_flag = False

                    for k in range(0, len(matrix_equality)):
                        temp_result = 0
                        for j in range(0, len(used_columns)):
                            temp_result += (matrix[k][used_columns[j]] * x[j])
                        if abs(temp_result - basis[k]) > 0.0000001:
                            limits_flag = False

                    for k in range(0, len(matrix_less)):
                        temp_result = 0
                        for j in range(0, len(used_columns)):
                            temp_result += (matrix[len(matrix_equality) + k][used_columns[j]] * x[j])
                        if temp_result > basis[len(matrix_equality) + k]:
                            limits_flag = False

                    for k in range(0, len(matrix_more)):
                        temp_result = 0
                        for j in range(0, len(used_columns)):
                            temp_result += (matrix[len(matrix_equality) + len(matrix_less) + k][used_columns[j]] * x[j])
                        if temp_result < basis[len(matrix_equality) + len(matrix_less) + k]:
                            limits_flag = False

                    if limits_flag:
                        temp_result = 0
                        for k in range(0, len(used_columns)):
                            temp_result += (function[used_columns[k]] * x[k])
                        if answer_result < temp_result:
                            answer_result = temp_result
                            answer_vector = x.copy()
                            saved_column_indices = used_columns
                except:
                    continue
    except StopIteration:
        reference_vector = [0] * len(function)
        for i in range(0, len(saved_column_indices)):
            reference_vector[saved_column_indices[i]] = answer_vector[i]
        j = 0
        result = [0] * n
        for i in range(0, n):
            if x_limits[i] == 0:
                result[i] = reference_vector[i] - reference_vector[n + j]
                j += 1
            else:
                result[i] = reference_vector[i]
        return result

def simplex_method(matrix, function, basis):
    extended_matrix = matrix
    for i in range(len(extended_matrix)):
        extended_matrix[i].append(basis[i])

    used_columns = [-1] * len(matrix)
    for i in range(0, len(function)):
        not_null_nums = 0
        first_num_row = -1
        for j in range(0, len(extended_matrix)):
            if extended_matrix[j][i] != 0:
                if first_num_row == -1:
                    first_num_row = j
                not_null_nums += 1
        if not_null_nums == 1:
            used_columns[first_num_row] = i

    for i in range(0, len(used_columns)):
        if used_columns[i] != -1:
            for j in range(0, len(extended_matrix[i])):
                if j != used_columns[i]:
                    extended_matrix[i][j] /= extended_matrix[i][used_columns[i]]
            extended_matrix[i][used_columns[i]] = 1

    for i in range(0, len(used_columns)):
        if used_columns[i] == -1:
            for j in range(0, len(function)):
                if j not in used_columns and extended_matrix[i][j] != 0:
                    for m in range(0, len(extended_matrix[i])):
                        if m != j:
                            extended_matrix[i][m] /= extended_matrix[i][j]
                    extended_matrix[i][j] = 1
                    for k in range(len(extended_matrix)):
                        if k != i:
                            div = extended_matrix[k][j] / extended_matrix[i][j]
                            for m in range(0, len(extended_matrix[k])):
                                extended_matrix[k][m] -= div * extended_matrix[i][m]
                    used_columns[i] = j
                    break
    print(np.array(extended_matrix))
    print(used_columns)

    row_max_b = 0
    for i in range(1, len(extended_matrix)):
        if abs(extended_matrix[i][-1]) >= abs(extended_matrix[row_max_b][-1]):
            row_max_b = i

    max_column = 0
    for i in range(1, len(extended_matrix[row_max_b]) - 1):
        if abs(extended_matrix[row_max_b][i]) >= abs(extended_matrix[row_max_b][max_column]):
            max_column = i

    for m in range(0, len(extended_matrix[row_max_b])):
        if m != max_column:
            extended_matrix[row_max_b][m] /= extended_matrix[row_max_b][max_column]
    extended_matrix[row_max_b][max_column] = 1

    for k in range(len(extended_matrix)):
        if k != row_max_b:
            div = extended_matrix[k][max_column] / extended_matrix[row_max_b][max_column]
            for m in range(0, len(extended_matrix[k])):
                extended_matrix[k][m] -= div * extended_matrix[row_max_b][m]

    used_columns[row_max_b] = max_column

    delta = []

    for i in range(0, len(function)):
        temp_result = 0
        for j in range(0, len(used_columns)):
            temp_result += function[used_columns[j]] * extended_matrix[j][i]
        temp_result -= function[i]
        delta.append(temp_result)

    temp_result = 0
    for j in range(0, len(used_columns)):
        temp_result += function[used_columns[j]] * extended_matrix[j][-1]
    temp_result -= function[-1]
    delta.append(temp_result)

    count = 0
    while not check_positive(delta):
        print("delta", count, delta)
        for i in range(0, len(extended_matrix)):
            print("s", count, extended_matrix[i])

        min_delta = 0
        for i in range(1, len(delta) - 1):
            if delta[i] < delta[min_delta]:
                min_delta = i

        simplex_relationships = []
        for i in range(0, len(extended_matrix)):
            if extended_matrix[i][min_delta] != 0:
                simplex_relationships.append(extended_matrix[i][-1] / extended_matrix[i][min_delta])
            else:
                simplex_relationships.append(-1)

        is_continue = False
        min_simplex_relationships = 0
        for i in range(0, len(simplex_relationships)):
            if (simplex_relationships[i] > 0 and not is_continue) or (
                    0 < simplex_relationships[i] < simplex_relationships[min_simplex_relationships]):
                min_simplex_relationships = i
                is_continue = True

        print("sr", count, simplex_relationships, min_simplex_relationships)
        if not is_continue:
            print("None")
            return None

        for m in range(0, len(extended_matrix[min_simplex_relationships])):
            if m != min_delta:
                extended_matrix[min_simplex_relationships][m] /= extended_matrix[min_simplex_relationships][min_delta]
        extended_matrix[min_simplex_relationships][min_delta] = 1

        for k in range(len(extended_matrix)):
            if k != min_simplex_relationships:
                div = extended_matrix[k][min_delta] / extended_matrix[min_simplex_relationships][min_delta]
                for m in range(0, len(extended_matrix[k])):
                    extended_matrix[k][m] -= div * extended_matrix[min_simplex_relationships][m]

        used_columns[min_simplex_relationships] = min_delta

        for i in range(0, len(extended_matrix)):
            print("!", count, extended_matrix[i])
        print(used_columns)

        delta = []

        for i in range(0, len(function)):
            temp_result = 0
            for j in range(0, len(used_columns)):
                temp_result += function[used_columns[j]] * extended_matrix[j][i]
            temp_result -= function[i]
            delta.append(temp_result)

        temp_result = 0
        for j in range(0, len(used_columns)):
            temp_result += function[used_columns[j]] * extended_matrix[j][-1]
        temp_result -= function[-1]
        delta.append(temp_result)
    print(used_columns, np.array(extended_matrix))


def check_positive(vector):
    answer = True
    for i in range(0, len(vector)):
        if (vector[i] < 0):
            answer = False
    return answer

def canonization(matrix_equality, matrix_more, matrix_less, x_limits, target_func, is_min):
    if is_min:
        function = np.copy(np.array(target_func) * -1)
    else:
        function = np.copy(np.array(target_func))

    matrix = matrix_equality + matrix_less + matrix_more

    basis = []
    for i in range(0, len(matrix)):
        basis.append(matrix[i][-1])

    rows = len(matrix)
    for i in range(rows):
        _ = matrix[i].pop(len(target_func))

    for i in range(len(x_limits)):
        if x_limits[i] < 0:
            for j in range(len(matrix)):
                matrix[j][i] *= -1
            function[i] *= -1
        if x_limits[i] == 0:
            for j in range(len(matrix)):
                matrix[j].append(-1 * matrix[j][i])
            function = np.append(function, [function[i] * -1])

    if (len(matrix_more) + len(matrix_less)) > 0:
        additional_matrix = []
        for i in range(0, len(matrix_equality)):
            additional_matrix.append([0] * (len(matrix_more) + len(matrix_less)))

        for i in range(0, len(matrix_less)):
            new_stroke = [0] * (len(matrix_more) + len(matrix_less))
            new_stroke[i] = 1
            additional_matrix.append(new_stroke)

        for i in range(0, len(matrix_more)):
            new_stroke = [0] * (len(matrix_more) + len(matrix_less))
            new_stroke[(len(matrix_less) + i)] = -1
            additional_matrix.append(new_stroke)

        function = np.append(function, [0] * (len(matrix_more) + len(matrix_less)))
        for i in range(0, len(matrix)):
            matrix[i] = matrix[i] + additional_matrix[i]

    return matrix, function, basis


# ==
A = [[-8, 3, 1, -2, -2, -2, 23], [-4, 2, 1, -1, -1, -1, 20], [-4, 2, 2, -1, -1, -1, 29]]

# >=
B = [[13, -3, -1, 3, 3, 3, -13]]

# <=
C = [[-23, 3, 1, -2, -3, -5, 17], [-18, 3, 1, -3, -3, -4, 12]]

F = [4, 6, 2, 6, 5, 3]

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

# 0 - not, 1 -> xi > 0, -1 -> xi < 0
xLimits = [1, 1, 1, 0, 0, 0]

if __name__ == '__main__':
    matrix, function, basis = canonization(A, B, C, xLimits, F, True)
    print(np.array(matrix))
    print(brute_force(matrix, function, basis, xLimits, A, B, C))
    simplex_method(matrix, function, basis)
