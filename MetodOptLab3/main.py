import eel

import logic.solve as logic

@eel.expose
def get_fibonacci_method_result(function):
    answer = []
    for i in [0.1, 0.01, 0.001]:
        answer.append([logic.fibonacciMethod(-3, 5, i, 0.01, logic.string_to_function(function)), logic.count_of_calls])
    return answer

@eel.expose
def get_golden_ratio_method_result(function):
    answer = []
    for i in [0.1, 0.01, 0.001]:
        print(function)
        answer.append([logic.golden_ratio(-3, 5, i, logic.string_to_function(function)), logic.count_of_calls])
    return answer


eel.init('web')

eel.start('index.html')

