import time
import eel
import numpy as np
import matplotlib.pyplot as plt

import logic.solve as logic

@eel.expose
def get_fibonacci_method_result(function, a, b, delta):
    answer = []
    for i in [0.1, 0.01, 0.001]:
        print(logic.fibonacciMethod(a, b, i, delta, logic.string_to_function(function)))
        x_p, R, d, S = logic.fibonacciMethod(a, b, i, delta, logic.string_to_function(function))
        answer.append([i, x_p, R, d, S])
        x = np.arange(a, b, 0.01)
        plt.clf()
        plt.xlabel("x")
        plt.ylabel("y")
        y = [logic.string_to_function(function)(element) for element in x]
        plt.plot(x, y)
        plt.errorbar(x_p, R, xerr= d, marker='o', linestyle='none',
        ecolor='k', elinewidth=0.8, capsize=4, capthick=1)
        plt.grid()
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)
        print(answer)
    return answer

@eel.expose
def get_golden_ratio_method_result(function, a, b):
    answer = []
    for i in [0.1, 0.01, 0.001]:
        x_p, R, d = logic.golden_ratio(a, b, i, logic.string_to_function(function))
        answer.append([i, x_p, R, d, logic.count_of_calls])
        x = np.arange(a, b, 0.01)
        plt.clf()
        plt.xlabel("x")
        plt.ylabel("y")
        y = [logic.string_to_function(function)(element) for element in x]
        plt.plot(x, y)
        plt.errorbar(x_p, R, xerr=d, marker='o', linestyle='none',
        ecolor='k', elinewidth=0.8, capsize=4, capthick=1)
        plt.grid()
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)
        print(answer)
    return answer


eel.init('web')

eel.start('index.html')

