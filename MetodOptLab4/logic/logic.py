import random
import matplotlib.pyplot as plt
import numpy as np

f = "((x - 1) ** 2) / (1) + ((y + 3) ** 2) / (0.4)"

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
    x_old_value = init_x
    y_old_value = init_y
    gradient_init = gradient(init_x, init_y, epsilon, function)
    x_new_value = init_x - step * gradient_init[0][0]
    y_new_value = init_y - step * gradient_init[1][0]
    while abs(function(x_old_value, y_old_value) - function(x_new_value, y_new_value)) > epsilon:
        x_old_value = x_new_value
        y_old_value = y_new_value
        gradient_new = gradient(x_new_value, y_new_value, epsilon, function)
        x_new_value = x_new_value - step * gradient_new[0][0]
        y_new_value = y_new_value - step * gradient_new[1][0]
        path[0].append(x_new_value)
        path[1].append(y_new_value)
    return x_new_value, y_new_value, path


def tuple2colvec(x: tuple) -> np.array:
    return np.array(x)[np.newaxis, ...].T


def colvec2tuple(x: np.array) -> tuple:
    return tuple(x.T[0])


def newton_method(init_x, epsilon, step, function):
    path = [[init_x[0]], [init_x[1]]]

    x = tuple2colvec(init_x)
    for i in range(0, 100):
        gradient_value = gradient(x[0][0], x[1][0], epsilon, function)
        hessian = [[derivative_xx(x[0][0], x[1][0], epsilon, function),
                    derivative_xy(x[0][0], x[1][0], epsilon, function)],
                   [derivative_yx(x[0][0], x[1][0], epsilon, function),
                    derivative_yy(x[0][0], x[1][0], epsilon, function)]]
        x_old_value = x

        x = x - step * (np.matmul(hessian, gradient_value))

        if is_not_improved(x_old_value, x, epsilon):
            break

        path[0].append(x[0][0])
        path[1].append(x[1][0])
    return path[0][-1], path[1][-1], path

def broyden_fletcher_goldfarb_shanno_method(init_x, epsilon, step, function):
    path = [[init_x[0]], [init_x[1]]]

    x = tuple2colvec(init_x)
    x_old_value = None
    for i in range(0, 100):
        gradient_value = gradient(x[0][0], x[1][0], epsilon, function)

        if x_old_value is None:
            hessian = np.identity(x.shape[0])
        else:
            s = x - x_old_value
            y = np.subtract(gradient_value, gradient_old_value)
            hessian = (np.identity(x.shape[0]) - (s @ y.T) / (y.T @ s)) @ hessian @ (np.identity(x.shape[0]) - (y @ s.T) / (y.T @ s)) + ((s @ s.T) / (y.T @ s))

        x_old_value = x

        x = x - step * (np.matmul(hessian, gradient_value))

        if is_not_improved(x_old_value, x, epsilon):
            break

        path[0].append(x[0][0])
        path[1].append(x[1][0])
        gradient_old_value = gradient_value
    return path[0][-1], path[1][-1], path


def is_not_improved(x_prev, x, epsilon):
    return np.abs(x_prev - x).sum() < epsilon


if __name__ == '__main__':
    math_function = string_to_function(f)

    x = np.linspace(-2, 4, 500)
    y = np.linspace(-5, 1, 500)
    X, Y = np.meshgrid(x, y)
    F = string_to_function(f)(X, Y)

    a_x, a_y, path = gradient_method(0, 0, 0.0001, 0.1, math_function)
    print(a_x, a_y)
    print(math_function(a_x, a_y))
    print(path)

    a_x, a_y, path = newton_method((0, 0), 0.0001, 0.01, math_function)
    print(a_x, a_y)
    print(math_function(a_x, a_y))
    print(path)

    a_x, a_y, path = broyden_fletcher_goldfarb_shanno_method((0, 0), 0.0001, 0.1, math_function)
    print(a_x, a_y)
    print(math_function(a_x, a_y))
    print(path)
