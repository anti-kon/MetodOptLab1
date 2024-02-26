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


def continue_solve(mark_in):  # проверка положительных оценок
    mark = np.copy(mark_in)
    mark = mark[1:]
    for i in mark:
        if i > 0:
            return True,
    return False


def get_mark(matrix, function, basis):  # вычисление оценки
    c_basis = []
    for i in basis:
        c_basis.append(function[i - 1])
    mark = np.dot(c_basis, matrix) - (np.append([0], function))
    print(mark)
    print('-----------------')
    return mark


def get_basis(matrix):  # получение базиса
    basis = []
    for i in range(len(matrix)):
        basis.append(matrix.shape[1] - len(matrix) + i)
    return basis


def add_additional_variables(matrix, function):  # добавление переменных к матрице и функции
    matrix = np.concatenate((matrix, np.eye(matrix.shape[0])), axis=1)
    function = np.append(function, matrix.shape[0] * [0])
    return matrix, function


def recount(matrix_in, index_input, index_output):  # пересчет мартрицы
    matrix = matrix_in.copy()
    k = matrix[index_output][index_input]
    matrix[index_output] /= k

    for i in range(len(matrix)):
        if i != index_output:
            matrix[i] -= matrix[i][index_input] * matrix[index_output]
    print(matrix)
    print('-----------------')
    return matrix


def get_index_input(mark):
    return np.argmax(mark)


def get_index_output(index_input, matrix_in):
    matrix = np.copy(matrix_in)
    p_0 = matrix[:, 0]
    p_i = matrix[:, index_input]

    p_i[p_i == 0] = -1  # exclude division by zero

    teta = p_0 / p_i
    teta = np.where(teta > 0, teta, np.inf)
    index_output = teta.argmin()

    if teta[index_output] == np.inf:
        raise Exception("Not solution")
    else:
        return index_output


def solve(matrix, function, basis):
    print(matrix)
    print("-----------------")
    mark = get_mark(matrix, function, basis)
    flag = continue_solve(mark)

    while flag:  # main loop

        index_input = get_index_input(mark)
        index_output = get_index_output(index_input, matrix)

        matrix = recount(matrix, index_input, index_output)

        basis[index_output] = index_input

        mark = get_mark(matrix, function, basis)
        flag = continue_solve(mark)

    return matrix, function, basis


def simplex_method(matrix, function, basis):
    matrix, function, basis = solve(matrix, function, basis)
    mark = get_mark(matrix, function, basis)

    p_0 = matrix[:, 0]

    x = np.zeros(len(C))

    for i in range(len(basis)):
        if (basis[i] - 1) < len(C):
            x[basis[i] - 1] = p_0[i]

    print("x = " + str(x))
    print("result = " + str(mark[0] * -1))


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
A = [[0, -2, 0, 1, 1, -3], [0, 0, 1, -2, 0, 2], [0, 0, 3, 0, 10, 12]]

# >=
B = [[1, 0, 1, 0, 0, -3], [0, 2, 0, 2, 0, 6]]

# <=
C = [[1, 3, 0, -1, 0, 5]]

F = [-2, 1, -1, 0, 1]

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
xLimits = [1, 1, 1, 0, 0]

if __name__ == '__main__':
    matrix, function, basis = canonization(A, B, C, xLimits, F, True)
    print(brute_force(matrix, function, basis, xLimits, A, B, C))