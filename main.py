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

    x_limits += [1] * (len(function) - len(x_limits))

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

def simplex_method(matrix, function, basis, x_limits, n):
    start_x_limit_size = len(x_limits)
    extended_matrix = matrix

    for i in range(len(extended_matrix)):
        extended_matrix[i].append(basis[i])

    extended_matrix.append([0] * len(function))
    for i in range(0, len(matrix[-1])):
        matrix[-1][i] = function[i] * -1
    extended_matrix[-1].append(0)

    used_columns = [-1] * (len(extended_matrix) - 1)
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

    for i in range(0, n):
        m = len(extended_matrix[i])
        for j in range(0, len(extended_matrix)):
            if j == i:
                extended_matrix[j].insert(m - (len(basis) - n) - 1, 1)
            else:
                extended_matrix[j].insert(m - (len(basis) - n) - 1, 0)

    while not check_positive(extended_matrix[-1]):
        min_column = -1

        for i in range(0, len(extended_matrix[-1]) - 1):
            if extended_matrix[-1][i] < 0:
                min_column = i
                break

        special_coff = [-1] * (len(extended_matrix) - 1)

        for i in range(0, len(extended_matrix)):
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

        used_columns[min_row] = min_column

    reference_vector = [0] * len(function)
    for i in range(0, len(extended_matrix) - 1):
        reference_vector[used_columns[i]] = extended_matrix[i][-1]
    j = 0
    result = [0] * start_x_limit_size
    for i in range(0, start_x_limit_size):
        if x_limits[i] == 0:
            result[i] = reference_vector[i] - reference_vector[n + j]
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

def get_double_task (matrix_equality, matrix_more, matrix_less, x_limits, target_func, is_min):
    is_double_min = not is_min
    if (is_min):
        for i in range(0, len(matrix_less)):
            for j in range(0, len(matrix_less[i])):
                matrix_less[i][j] *= -1
        for i in range(0, len(x_limits)):
            if x_limits[i] < 0:
                x_limits[i] *= -1
                for j in range(0, len(matrix_less)):
                    matrix_less[j][i] *= -1
                for j in range(0, len(matrix_more)):
                    matrix_more[j][i] *= -1
                for j in range(0, len(matrix_equality)):
                    matrix_equality[j][i] *= -1
    else:
        for i in range(0, len(matrix_more)):
            for j in range(0, len(matrix_more[i])):
                matrix_more[i][j] *= -1
        for i in range(0, len(x_limits)):
            if x_limits[i] > 0:
                x_limits[i] *= -1
                for j in range(0, len(matrix_less)):
                    matrix_less[j][i] *= -1
                for j in range(0, len(matrix_more)):
                    matrix_more[j][i] *= -1
                for j in range(0, len(matrix_equality)):
                    matrix_equality[j][i] *= -1

    if (len(matrix_equality) != 0):
        new_matrix = np.array(matrix_equality).transpose()
        if (len(matrix_more) != 0):
            new_matrix = new_matrix + np.array(matrix_more).transpose()
        if (len(matrix_less) != 0):
            new_matrix = new_matrix + np.array(matrix_less).transpose()
    elif (len(matrix_more) != 0):
        new_matrix = np.array(matrix_more).transpose()
        if (len(matrix_less) != 0):
            new_matrix = new_matrix + np.array(matrix_less).transpose()
    else:
        new_matrix = np.array(matrix_less).transpose()

    new_equality = []
    new_more = []
    new_less = []

    for i in range(0, len(target_func)):
        if target_func[i] > 0:
            new_equality.append([0] * (len(new_matrix[i]) + 1))
            for j in range(0, len(new_matrix[i])):
                new_equality[-1][j] = new_matrix[i][j]
            new_equality[-1][-1] = target_func[i]
        else:
            if is_min:
                new_less.append([0] * (len(new_matrix[i]) + 1))
                for j in range(0, len(new_matrix[i])):
                    new_less[-1][j] = new_matrix[i][j]
                new_less[-1][-1] = target_func[i]
            else:
                new_more.append([0] * (len(new_matrix[i]) + 1))
                for j in range(0, len(new_matrix[i])):
                    new_more[-1][j] = new_matrix[i][j]
                new_more[-1][-1] = target_func[i]

    new_target_func = []
    for i in range(0, len(new_matrix[-1])):
        new_target_func.append(new_matrix[-1][i])

    new_x_limits = [0] * len(new_target_func)
    for i in range(0, len(matrix_less) + len(matrix_more)):
        new_x_limits[len(matrix_equality) - 1 + i] = 1

    return new_equality, new_more, new_less, new_target_func, new_x_limits, is_double_min
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

# ==
A = []

B = []

C = [[6, 3, 1, 4, 252], [2, 4, 5, 1, 144], [1, 2, 4, 3, 80]]

F = [48, 33, 16, 22]

# 0 - not, 1 -> xi > 0, -1 -> xi < 0
xLimits = [1, 1, 1, 1]

if __name__ == '__main__':
    input_equality = []
    input_inequality_less = []
    input_inequality_more = []
    x_count = 0
    restrictions_count = 0
    input_target_func = []
    input_is_min = False

    input_check = False
    while not input_check:
        x_count = input("Введите количество переменных: ")
        if int(x_count) > 0:
            x_count = int(x_count)
            input_check = True
        else:
            print("Введено недопустимое значение")

    input_check = False
    while not input_check:
        restrictions_count = input("Введите количество ограничений: ")
        if int(restrictions_count) > 0:
            restrictions_count = int(restrictions_count)
            input_check = True
        else:
            print("Введено недопустимое значение")


    for i in range(0, int(restrictions_count)):
        array_type = 0

        input_check = False
        while not input_check:
            i_restriction = input("Выберите знак " + str((i+1)) + "-ого ограничения (-1 — ≤, 0 — =, 1 — ≥): ")
            if int(i_restriction) == -1 or int(i_restriction) == 0 or int(i_restriction) == 1:
                array_type = int(i_restriction)
                input_check = True
            else:
                print("Введено недопустимое значение")

        row = [0] * (x_count + 1)
        i_restriction = input("Введите вектор (левую часть) " + str((i + 1)) + "-ого ограничения (через пробел): ")
        restriction_vector = i_restriction.split()
        for j in range(0, len(restriction_vector)):
            if j > x_count:
                break
            row[j] = int(restriction_vector[j])

        i_restriction = input("Введите правую часть " + str((i + 1)) + "-ого ограничения: ")
        row[-1] = int(i_restriction)

        print("Было введено ограничение: ", end='')
        for j in range(0, len(row) - 1):
            if row[j] > 0:
                print("+", row[j], "* x" + str(j + 1), end=' ')
            if row[j] < 0:
                print("-", abs(row[j]), "* x" + str(j + 1), end=' ')

        if array_type == 0:
            input_equality.append(row)
            print("=", input_equality[-1][-1])
        elif array_type == -1:
            input_inequality_less.append(row)
            print("≤", input_inequality_less[-1][-1])
        elif array_type == 1:
            input_inequality_more.append(row)
            print("≥", input_inequality_more[-1][-1])

    input_target_func = [0] * x_count
    target_func_string = input("Введите вектор функции цели (через пробел): ")
    target_func_array = target_func_string.split()
    for j in range(0, len(target_func_array)):
        input_target_func[j] = int(target_func_array[j])

    input_check = False
    while not input_check:
        is_min_string = input("Выберите тип задачи (0 — min, 1 — max) ")
        if int(is_min_string) == 0:
            input_is_min = True
            input_check = True
        elif int(is_min_string) == 1:
            input_is_min = False
            input_check = True
        else:
            print("Введено недопустимое значение")

    x_limits = [0] * x_count
    x_limits_string = input("Введите вектор ограничений на знак, через пробел (-1 — xᵢ ≤ 0, 0 — нет ограничения для "
                            "xᵢ, -1 — xᵢ ≥ 0): ")
    x_limits_array = x_limits_string.split()
    for j in range(0, len(x_limits_array)):
        x_limits[j] = int(x_limits_array[j])

    print("Было введено условие:")
    print("\tФункция цели: ")
    print("\t\tF(X) = ", end='')
    for j in range(0, len(input_target_func)):
        if input_target_func[j] > 0:
            print("+", input_target_func[j], "* x" + str(j + 1), end=' ')
        if input_target_func[j] < 0:
            print("-", abs(input_target_func[j]), "* x" + str(j + 1), end=' ')
    print("-> ", end='')
    if input_is_min:
        print("min")
    else:
        print("max")
    print("\tСистема ограничений: ")
    for i in range(0, len(input_equality)):
        print("\t\t", end='')
        for k in range(0, len(input_equality[i]) - 1):
            if input_equality[i][k] > 0:
                print("+", input_equality[i][k], "* x" + str(j + 1), end=' ')
            if input_equality[i][k] < 0:
                print("-", abs(input_equality[i][k]), "* x" + str(j + 1), end=' ')
        print("=", input_equality[i][-1])
    for i in range(0, len(input_inequality_less)):
        print("\t\t", end='')
        for k in range(0, len(input_inequality_less[i]) - 1):
            if input_inequality_less[i][k] > 0:
                print("+", input_inequality_less[i][k], "* x" + str(j + 1), end=' ')
            if input_inequality_less[i][k] < 0:
                print("-", abs(input_inequality_less[i][k]), "* x" + str(j + 1), end=' ')
        print("≤", input_inequality_less[i][-1])
    for i in range(0, len(input_inequality_more)):
        print("\t\t", end='')
        for k in range(0, len(input_inequality_more[i]) - 1):
            if input_inequality_more[i][k] > 0:
                print("+", input_inequality_more[i][k], "* x" + str(j + 1), end=' ')
            if input_inequality_more[i][k] < 0:
                print("-", abs(input_inequality_more[i][k]), "* x" + str(j + 1), end=' ')
        print("≥", input_inequality_more[i][-1])
    for i in range(0, len(x_limits)):
        if x_limits[i] > 0:
            print("\t\tx" + str(i + 1) + " ≥ 0")
        if x_limits[i] < 0:
            print("\t\tx" + str(i + 1) + " ≤ 0")

    remember_a = np.array(input_equality)
    remember_b = np.array(input_inequality_more)
    remember_c = np.array(input_inequality_less)
    matrix, function, basis = canonization(input_equality.copy(), input_inequality_more.copy(), input_inequality_less.copy(), x_limits.copy(), input_target_func.copy(), input_is_min)
    remember_f = input_target_func.copy()
    print("Метод перебора крайних точек: ", brute_force(matrix.copy(), function.copy(), basis.copy(), x_limits.copy(), input_equality.copy(), input_inequality_more.copy(), input_inequality_less.copy()))
    print("Симплекс-метод: ", simplex_method(matrix.copy(), function.copy(), basis.copy(), x_limits.copy(), len(input_equality)))
    new_equality, new_more, new_less, new_target_func, new_x_limits, new_is_min = get_double_task(remember_a, remember_b, remember_c, x_limits.copy(), remember_f.copy(), False)
    print("Двойственная задача: ")
    print("\tФункция цели: ")
    print("\t\tF(X) = ", end='')
    for j in range(0, len(new_target_func)):
        if new_target_func[j] > 0:
            print("+", new_target_func[j], "* x" + str(j + 1), end=' ')
        if new_target_func[j] < 0:
            print("-", abs(new_target_func[j]), "* x" + str(j + 1), end=' ')
    print("-> ", end='')
    if new_is_min:
        print("min")
    else:
        print("max")
    print("\tСистема ограничений: ")
    for i in range(0, len(new_equality)):
        print("\t\t", end='')
        for k in range(0, len(new_equality[i]) - 1):
            if new_equality[i][k] > 0:
                print("+", new_equality[i][k], "* x" + str(j + 1), end=' ')
            if new_equality[i][k] < 0:
                print("-", abs(new_equality[i][k]), "* x" + str(j + 1), end=' ')
        print("=", new_equality[i][-1])
    for i in range(0, len(new_less)):
        print("\t\t", end='')
        for k in range(0, len(new_less[i]) - 1):
            if new_less[i][k] > 0:
                print("+", new_less[i][k], "* x" + str(j + 1), end=' ')
            if new_less[i][k] < 0:
                print("-", abs(new_less[i][k]), "* x" + str(j + 1), end=' ')
        print("≤", new_less[i][-1])
    for i in range(0, len(new_more)):
        print("\t\t", end='')
        for k in range(0, len(new_more[i]) - 1):
            if new_more[i][k] > 0:
                print("+", new_more[i][k], "* x" + str(j + 1), end=' ')
            if new_more[i][k] < 0:
                print("-", abs(new_more[i][k]), "* x" + str(j + 1), end=' ')
        print("≥", new_more[i][-1])
    for i in range(0, len(new_x_limits)):
        if new_x_limits[i] > 0:
            print("\t\tx" + str(i + 1) + " ≥ 0")
        if new_x_limits[i] < 0:
            print("\t\tx" + str(i + 1) + " ≤ 0")

