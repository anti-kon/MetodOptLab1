import math
from math import cos
from math import sin
from math import tan
from math import acos
from math import asin
from math import atan
from math import e
from math import log
from math import pi
from math import gamma

def factorial(a):
    return gamma(a+1)

def acot(a):
    return 1 / atan(a)


def ln(a):
    return log(a)


def cot(a):
    return 1 / tan(a)


def sqrt(a, b):
    return a ** (1 / b)


count_of_calls = 0
func = [0, 2, 1, 4, 17, 21] # коэффициенты задаются, начиная с коэффициента при нулевой степени

# def calcFunc(arg: float):
#     global count_of_calls, func
#
#     count_of_calls += 1
#     result = 0
#     for i in range(len(func)):
#         result += func[i] * arg ** i
#
#     return result


def golden_ratio(a: float, b: float, epsilon: float, function):
    global count_of_calls
    count_of_calls = 0

    x1 = a + 0.382 * (b - a)
    x2 = b - 0.382 * (b - a)
    A = function(x1)
    B = function(x2)
    count_of_calls += 2
    while True:
        if (A < B):
            b = x2
            if (b - a) > epsilon:
                x2 = x1
                B = A
                x1 = a + 0.382 * (b - a)
                A = function(x1)
                count_of_calls += 1
            else:
               x = (a + b) / 2
               R = function(x)
               count_of_calls += 1
               return x, R, (b - a)
        else:
            a = x1
            if (b - a) > epsilon:
                x1 = x2
                A = B
                x2 = b - 0.382 * (b - a)
                B = function(x2)
                count_of_calls += 1
            else:
                x = (a + b) / 2
                R = function(x)
                return x, R, (b - a)

def kMoreThanSMinusOne(a: float, b: float, epsilon: float, delta: float, x1: float, x2: float, A: float, B: float, function):
    global count_of_calls
    x2 = x1 + delta
    B = function(x2)
    count_of_calls += 1
    if A < B:
        b = x1
    else:
        a = x1
    x = (a + b) / 2
    R = function(x)
    return x, R, b - a

def fibonacciMethod(a: float, b: float, epsilon: float, delta: float, function):
    global count_of_calls
    count_of_calls = -2

    fibonacci_numbers = [1, 1]
    N = (b - a) / epsilon
    i = 2
    while True:
        fibonacci_numbers.append(fibonacci_numbers[i - 1] + fibonacci_numbers[i - 2])
        if N < fibonacci_numbers[i]:
            break
        else:
            i += 1
    S = i
    k = 1
    l = (b - a) / fibonacci_numbers[S]
    x1 = a + l * fibonacci_numbers[S - 2]
    x2 = b - l * fibonacci_numbers[S - 2]
    A = function(x1)
    B = function(x2)
    count_of_calls += 2

    while True:
        if A < B:
            b = x2
            k += 1
            if k == S - 1:
                x, R, d = kMoreThanSMinusOne(a, b, epsilon, delta, x1, x2, A, B, function)
                return x, R, d, S
            else:
                x2 = x1
                B = A
                x1 = a + l * fibonacci_numbers[S - 1 - k]
                A = function(x1)
                count_of_calls += 1
        else:
            a = x1
            k += 1
            if k == S - 1:
                x, R, d = kMoreThanSMinusOne(a, b, epsilon, delta, x1, x2, A, B, function)
                return x, R, d, S
            else:
                x1 = x2
                A = B
                x2 = b - l * fibonacci_numbers[S - 1 - k]
                B = function(x2)
                count_of_calls += 1

def string_to_function(expression):
    def function(x):
        return eval(expression)
    return function

if __name__ == '__main__':
    lambda_func = input(str())
    a = 5
    b = 8
    for i in [0.001, 0.01, 0.1]:
        print("golden ratio for", i, ":", golden_ratio(a, b, i, string_to_function(lambda_func)), count_of_calls)
        print("fibonacci for", i, ":", fibonacciMethod(a, b, i, 0.00001, string_to_function(lambda_func)), count_of_calls)