import matplotlib.pyplot as plt
from scipy.optimize import line_search
import numpy as np
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


f = "cos(x + y) + x ** 2 + y ** 2"

# 2 + 12 x * 2  0   | (2 + 12 x^2 - l) (8) = 0 (l = 8) l = 2 + 12 x^2
# 0 8 - l              |
def string_to_function(expression):
    def function(x, y):
        return eval(expression)

    return function


def derivative_x(x, y, increment, function):
    return (function(x + increment, y) - function(x, y)) / increment


def derivative_y(x, y, increment, function):
    return (function(x, y + increment) - function(x, y)) / increment


def gradient(x, y, epsilon, function):
    return [[derivative_x(x, y, epsilon, function)], [derivative_y(x, y, epsilon, function)]]


def derivative_xx(x, y, increment, function):
    return (function(x + increment, y) - 2 * function(x, y) + function(x - increment, y)) / (increment ** 2)


def derivative_xy(x, y, increment, function):
    return (derivative_y(x + increment, y, increment, function) - derivative_y(x, y, increment, function)) / increment


def derivative_yx(x, y, increment, function):
    return (derivative_x(x, y + increment, increment, function) - derivative_x(x, y, increment, function)) / increment


def derivative_yy(x, y, increment, function):
    return (function(x, y + increment) - 2 * function(x, y) + function(x, y - increment)) / (increment ** 2)


def gradient_method(init_x, init_y, epsilon, step, function):
    path = [[init_x], [init_y]]
    gradient_init = gradient(init_x, init_y, epsilon, function)
    x_new_value = init_x - step * gradient_init[0][0]
    y_new_value = init_y - step * gradient_init[1][0]
    k = 0
    while True:
        k+=1
        x_old_value = x_new_value
        y_old_value = y_new_value
        gradient_new = gradient(x_new_value, y_new_value, epsilon, function)
        x_new_value = x_new_value - step * gradient_new[0][0]
        y_new_value = y_new_value - step * gradient_new[1][0]
        path[0].append(x_new_value)
        path[1].append(y_new_value)
        if is_not_improved(gradient(x_old_value, y_old_value, epsilon, function),
                           gradient(x_new_value, y_new_value, epsilon, function), epsilon):
            break
    return x_new_value, y_new_value, path, k


def tuple2colvec(x: tuple) -> np.array:
    return np.array(x)[np.newaxis, ...].T


def colvec2tuple(x: np.array) -> tuple:
    return tuple(x.T[0])


def newton_method(init_x, epsilon, delta, function):
    path = [[init_x[0]], [init_x[1]]]
    step = 1
    x = tuple2colvec(init_x)
    k = 0
    while True:
        k += 1
        gradient_value = gradient(x[0][0], x[1][0], epsilon, function)  # grad x k-1
        hessian = [[derivative_xx(x[0][0], x[1][0], epsilon, function),
                    derivative_xy(x[0][0], x[1][0], epsilon, function)],
                   [derivative_yx(x[0][0], x[1][0], epsilon, function),
                    derivative_yy(x[0][0], x[1][0], epsilon, function)]]
        hessian = np.linalg.inv(hessian)
        x_old_value = x

        x = x - step*(np.matmul(hessian, gradient_value))  # x k = x(k-1) - ak * Hk^-1 * G(k-1)

        if is_not_improved(gradient(x_old_value[0][0], x_old_value[1][0], epsilon, function),
                           gradient(x[0][0], x[1][0], epsilon, function), epsilon):
            break

        path[0].append(x[0][0])
        path[1].append(x[1][0])

        while (function(x[0][0], x[1][0]) - function(x_old_value[0][0], x_old_value[1][0]) >
               -(0.1 * step * pow(np.abs(gradient(x_old_value[0][0], x_old_value[1][0], epsilon, function)).sum(), 2))):
            step *= delta

    return path[0][-1], path[1][-1], path, k


def broyden_fletcher_goldfarb_shanno_method(init_x, epsilon, function):
    path = [[init_x[0]], [init_x[1]]]

    def func_obj(x):
        return function(x[0], x[1])

    def gradient_obj(x):
        gradient_value = gradient(x[0], x[1], epsilon, function)
        return [gradient_value[0][0], gradient_value[1][0]]

    x = tuple2colvec(init_x)
    x_old_value = None
    k = 0
    while True:
        k += 1
        gradient_value = gradient(x[0][0], x[1][0], epsilon, function)
        if x_old_value is None:
            hessian = np.identity(x.shape[0])
        else:
            s = x - x_old_value
            y = np.subtract(gradient_value, gradient_old_value)
            hessian = (np.identity(x.shape[0]) - (s @ y.T) / (y.T @ s)) @ hessian @ (
                        np.identity(x.shape[0]) - (y @ s.T) / (y.T @ s)) + ((s @ s.T) / (y.T @ s))

        x_old_value = x

        delta = -hessian.dot([gradient_obj(x)[0][0], gradient_obj(x)[1][0]])
        step = line_search(f=func_obj, myfprime=gradient_obj, xk=np.array([x[0][0], x[1][0]]),
                           pk=delta, c1=0.0001, c2=0.9)[0]
        if step is None:
            break
        x = x - float(step) * (np.matmul(hessian, gradient_value))

        if is_not_improved(gradient(x_old_value[0][0], x_old_value[1][0], epsilon, function),
                           gradient(x[0][0], x[1][0], epsilon, function), epsilon):
            break

        path[0].append(x[0][0])
        path[1].append(x[1][0])
        gradient_old_value = gradient_value

    return path[0][-1], path[1][-1], path, k


def is_not_improved(x_prev, x, epsilon):
    return np.abs(np.subtract(x_prev, x)).sum() < epsilon


if __name__ == '__main__':
    math_function = string_to_function(f)

    a_x, a_y, path, k = newton_method((-1, -1), 0.001, 0.8, math_function)
    print(a_x, a_y, math_function(a_x, a_y))
    x = np.linspace(-1.5, 0.5, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    F = cos(X + Y)
    print(math_function(X, Y))

    fig, ax = plt.subplots()
    ax.grid(True)
    for i in range(0, len(path[0]) - 1):
        ax.contour(X, Y, F - math_function(path[0][i], path[1][i]), levels=[0])
        plt.plot([path[0][i], path[0][i + 1]], [path[1][i], path[1][i + 1]], c='r')
    ax.contour(X, Y, F - math_function(path[-1][0], path[-1][1]), levels=[0])
    ax.plot(a_x, a_y, 'ro')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Newton methods")

    fig, ax = plt.subplots()
    error = []
    for i in range(0, len(path[0])):
        error.append(np.abs(np.subtract(math_function(a_x, a_y), math_function(path[0][i], path[1][i]))).sum())
    line1, = ax.plot(range(1, len(error) + 1), error, label='Newton method')

    a_x, a_y, path, k = broyden_fletcher_goldfarb_shanno_method((-1, -1), 0.001, math_function)
    print(a_x, a_y, math_function(a_x, a_y))
    X, Y = np.meshgrid(x, y)
    F = math_function(X, Y)

    error = []
    for i in range(0, len(path[0]) - 1):
        error.append(np.abs(np.subtract(math_function(a_x, a_y), math_function(path[0][i], path[1][i]))).sum())
    line2, = ax.plot(range(1, len(error) + 1), error, label='BFGS method')

    ax.grid(True)
    plt.title("Comparison of the convergence rate of Newton and BFGS methods")
    ax.legend(handles=[line1, line2])
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.yscale("log")
    plt.xlim(1, 10)
    plt.ylim(0.001)

    fig, ax = plt.subplots()
    ax.grid(True)
    for i in range(0, len(path[0]) - 1):
        ax.contour(X, Y, F - math_function(path[0][i], path[1][i]), levels=[0])
        plt.plot([path[0][i], path[0][i + 1]], [path[1][i], path[1][i + 1]], c='r')
    ax.contour(X, Y, F - math_function(path[-1][0], path[-1][1]), levels=[0])
    ax.plot(a_x, a_y, 'ro')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("BFGS methods")


    plt.show()