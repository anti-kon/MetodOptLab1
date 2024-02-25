import math

import numpy as np


def gauss(m):
    # eliminate columns
    for col in range(len(m[0])):
        for row in range(col + 1, len(m)):
            r = [(rowValue * (-(m[row][col] / m[col][col]))) for rowValue in m[col]]
            m[row] = [sum(pair) for pair in zip(m[row], r)]
    # now backsolve by substitution
    ans = []
    m.reverse()  # makes it easier to backsolve
    for sol in range(len(m)):
        if sol == 0:
            ans.append(m[sol][-1] / m[sol][-2])
        else:
            inner = 0
            # substitute in all known coefficients
            for x in range(sol):
                inner += (ans[x] * m[sol][-2 - x])
            # the equation is now reduced to ax + b = c form
            # solve with (c - b) / a
            ans.append((m[sol][-1] - inner) / m[sol][-sol - 2])
    ans.reverse()
    return ans


def brut(matrixEquality, matrixLess, matrixMore, xLimits, targetFunc, isMin):
    A = matrixEquality + matrixLess + matrixMore
    if isMin:
        answerResult = math.inf
    else:
        answerResult = -math.inf
    answerVector = []

    for i in range (0, len(A)):
        B = []
        for j in range (0, len(A)):
            if i != j:
                B.append(A[j])
        for i in range (0, len(B)):
            if B[i][i] == 0.0:
                for j in range(0, len(B)):
                    if B[j][i] != 0.0 and B[i][j] != 0.0:
                        tempStroke = B[j]
                        B[j] = B[i]
                        B[i] = tempStroke
        x = gauss(B)
        limitsFlag = True

        for i in range(0, len(xLimits)):
            if xLimits[i] == 1:
                if x[i] < 0:
                    limitsFlag = False
            elif xLimits == -1:
                if x[i] > 0:
                    limitsFlag = False

        for i in range(0, len(matrixEquality)):
            tempResult = 0
            for j in range(0, len(matrixEquality[i]) - 1):
                tempResult += (matrixEquality[i][j] * x[j])
            if (tempResult != matrixEquality[i][-1]):
                limitsFlag = False

        for i in range(0, len(matrixLess)):
            tempResult = 0
            for j in range(0, len(matrixLess[i]) - 1):
                tempResult += (matrixLess[i][j] * x[j])
            if (tempResult > matrixLess[i][-1]):
                limitsFlag = False

        for i in range(0, len(matrixMore)):
            tempResult = 0
            for j in range(0, len(matrixMore[i]) - 1):
                tempResult += (matrixMore[i][j] * x[j])
            if (tempResult < matrixMore[i][-1]):
                limitsFlag = False

        if limitsFlag:
            tempResult = 0
            for i in range(0, len(targetFunc)):
                tempResult += (targetFunc[i] * x[i])
            if isMin:
                if answerResult > tempResult:
                    answerResult = tempResult
                    answerVector = x.copy()
            else:
                if answerResult < tempResult:
                    answerResult = tempResult
                    answerVector = x.copy()

    print(answerResult, answerVector)


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


def canonization(matrixEquality, matrixLess, matrixMore, xLimits, targetFunc, isMin):
    if isMin:
        function = np.copy(np.array(targetFunc) * -1)
    else:
        function = np.copy(np.array(targetFunc))

    matrix = matrixEquality + matrixLess + matrixMore
    rows = len(matrix)
    for i in range(rows):
        _ = matrix[i].pop(len(targetFunc))

    print(function, matrix)
    for i in range(len(xLimits)):
        if (xLimits[i] < 0):
            for j in range(len(matrix)):
                matrix[j][i] *= -1
            function[i] *= -1
        if (xLimits[i] == 0):
            for j in range(len(matrix)):
                matrix[j].append(-1 * matrix[j][i])
            function = np.append(function, [function[i] * -1])

    print(function, matrix)


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



# ==
A = [[0, -2, 0, 1, 1, -3], [0, 0, 1, -2, 0, 2], [0, 0, 3, 0, 10, 12]]

# >=
B = [[1, 0, 1, 0, 0, -3], [0, 2, 0, 2, 0, 6]]

# <=
C = [[1, 3, 0, -1, 0, 5]]

F = [-2, 1, -1, 0, 1]

# 0 - not, 1 -> xi > 0, -1 -> xi < 0
xLimits = [1, 1, 1, 0, 0]

if __name__ == '__main__':
    brut(A, C, B, xLimits, F, True)
    canonization(A, B, C, xLimits, F, True)
