import eel

import logic.solve as logic

@eel.expose
def get_fibonacci_method():
    return logic.fibonacciMethod(-3, 5, 0.1, 0.01, logic.string_to_function(logic.lambda_func)), logic.count_of_calls

@eel.expose
def get_potentials_method_result():
    return logic.golden_ratio(-3, 5, 0.01, logic.string_to_function(logic.lambda_func)), logic.count_of_calls


eel.init('web')

eel.start('index.html')

