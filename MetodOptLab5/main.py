import time

import eel
import numpy as np
from matplotlib import pyplot as plt

from logic.logic import zeutendijk_method
import math
from numpy import cos
from numpy import sin
from numpy import tan
from math import acos
from math import asin
from math import atan
from numpy import e
from numpy import log
from numpy import pi
from math import gamma


def factorial(a):
    return gamma(a + 1)


def acot(a):
    return 1 / atan(a)


def ln(a):
    return log(a)


def cot(a):
    return 1 / tan(a)


def sqrt(a, b):
    return a ** (1 / b)


def string_to_function(expression):
    def function(X):
        args = {}
        for index in range(0, len(X)):
            args[('x' + str(index + 1))] = X[index]
        return eval(expression, globals(), args)

    return function


def string_to_function_array(expression_array):
    functions_array = []
    for index in range(0, len(expression_array)):
        functions_array.append(string_to_function(expression_array[index]))

    def function(X):
        function_results = []
        for index in range(0, len(functions_array)):
            function_results.append(functions_array[index](X))
        return np.array(function_results)

    return function

@eel.expose
def get_zeutendijk_method_possible_directions_result(function_equation, restrictions, signRestrictions,
                                                     lambdaValue, epsilon, n):
    print(lambdaValue, epsilon)
    func = string_to_function(function_equation)
    restrictions_equations = []
    for restriction in restrictions:
        restrictions_equations.append(restriction['equation'] + " - " + str(restriction['value']))
    # for sign_restriction in signRestrictions:
    #     restrictions_equations.append('-' + sign_restriction['equation'])
    arr = string_to_function_array(restrictions_equations)
    history, iteration = zeutendijk_method(n, func, arr, epsilon, lambdaValue)
    result = []
    result.append([])
    for i in range(0, len(history)):
        result[0].append([])
        result[0][-1].append(i)
        for j in range(0, len(history[i])):
            result[0][-1].append(history[i][j])
    result.append(iteration)
    result.append(result[0][-1])
    result.append(func(history[-1]))
    result.append([])
    for i in range(0, len(history) - 1):
        result[-1].append(abs(func(history[-1]) - func(history[i])))

    fig, ax = plt.subplots()
    ax.grid(True)
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.yscale("log")
    ax.plot(range(1, len(result[-1]) + 1), result[-1])
    plt.title("Сonvergence of Zeutendijk's method")
    name = "image/image" + str(time.time()) + ".png"
    plt.savefig("web/" + name, transparent=True)
    result.append(name)
    return result


eel.init('web')

eel.start('index.html')

