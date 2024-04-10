import time
import eel
import numpy as np
import matplotlib.pyplot as plt
import logic.solve as logic
@eel.expose
def get_gradient_method_result(function_str, x_input, y_input, x_min, x_max, y_min, y_max, x_split, y_split, step):
    answer = []
    for i in [0.1, 0.01, 0.001]:
        a_x, a_y, path, k = logic.gradient_method(x_input, y_input, i, step, logic.string_to_function(function_str))
        function = logic.string_to_function(function_str)
        answer.append([i, a_x, a_y, function(a_x, a_y)])
        print(a_x, a_y, function(a_x, a_y))
        x = np.linspace(x_min, x_max, x_split)
        y = np.linspace(y_min, y_max, y_split)
        X, Y = np.meshgrid(x, y)
        F = function(X, Y)

        fig, ax = plt.subplots()
        plt.grid(True)
        for i in range(0, len(path[0]) - 1):
            ax.contour(X, Y, F - function(path[0][i], path[1][i]), levels=[0])
            plt.plot([path[0][i], path[0][i + 1]], [path[1][i], path[1][i + 1]], c='r')
        ax.contour(X, Y, F - function(path[-1][0], path[-1][1]), levels=[0])
        ax.plot(a_x, a_y, 'ro')
        plt.xlabel("x")
        plt.ylabel("y")
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)

        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(a_x, a_y, function(a_x, a_y), color='red')
        ax.plot_surface(X, Y, F, rstride=5, cstride=5, alpha=0.7)
        for i in range(0, len(path[0]) - 1):
            ax.plot([path[0][i], path[0][i + 1]], [path[1][i], path[1][i + 1]],
                    [function(path[0][i], path[1][i]),
                     function(path[0][i + 1], path[1][i + 1])], c='r')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)

        fig, ax = plt.subplots()
        error = []
        plt.grid(True)
        for i in range(0, len(path[0])):
            error.append(abs(function(a_x, a_y) - function(path[0][i], path[1][i])))
        ax.plot(range(1, len(error) + 1), error)
        plt.xlabel("iteration")
        plt.ylabel("error")
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)
        plt.close('all')
        answer[-1].append(k)
    return answer

@eel.expose
def get_broyden_fletcher_goldfarb_shanno_method_result(function_str, x_input, y_input, x_min, x_max, y_min, y_max, x_split, y_split):
    answer = []
    for i in [0.1, 0.01, 0.001]:
        a_x, a_y, path, k = logic.broyden_fletcher_goldfarb_shanno_method((x_input, y_input), i, logic.string_to_function(function_str))
        function = logic.string_to_function(function_str)
        answer.append([i, a_x, a_y, function(a_x, a_y)])
        print(a_x, a_y, function(a_x, a_y))
        x = np.linspace(x_min, x_max, x_split)
        y = np.linspace(y_min, y_max, y_split)
        X, Y = np.meshgrid(x, y)
        F = function(X, Y)

        fig, ax = plt.subplots()
        plt.grid(True)
        for i in range(0, len(path[0]) - 1):
            ax.contour(X, Y, F - function(path[0][i], path[1][i]), levels=[0])
            plt.plot([path[0][i], path[0][i + 1]], [path[1][i], path[1][i + 1]], c='r')
        ax.contour(X, Y, F - function(path[-1][0], path[-1][1]), levels=[0])
        ax.plot(a_x, a_y, 'ro')
        plt.xlabel("x")
        plt.ylabel("y")
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)

        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(a_x, a_y, function(a_x, a_y), color='red')
        ax.plot_surface(X, Y, F, rstride=5, cstride=5, alpha=0.7)
        for i in range(0, len(path[0]) - 1):
            ax.plot([path[0][i], path[0][i + 1]], [path[1][i], path[1][i + 1]],
                    [function(path[0][i], path[1][i]),
                     function(path[0][i + 1], path[1][i + 1])], c='r')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)

        fig, ax = plt.subplots()
        error = []
        plt.grid(True)
        for i in range(0, len(path[0])):
            error.append(abs(function(a_x, a_y) - function(path[0][i], path[1][i])))
        ax.plot(range(1, len(error) + 1), error)
        plt.xlabel("iteration")
        plt.ylabel("error")
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)
        plt.close('all')
        answer[-1].append(k)
    return answer

@eel.expose
def get_newton_method_result(function_str, x_input, y_input, x_min, x_max, y_min, y_max, x_split, y_split, delta):
    answer = []
    for i in [0.1, 0.01, 0.001]:
        a_x, a_y, path, k = logic.newton_method((x_input, y_input), i, delta, logic.string_to_function(function_str))
        function = logic.string_to_function(function_str)
        answer.append([i, a_x, a_y, function(a_x, a_y)])
        print(a_x, a_y, function(a_x, a_y))
        x = np.linspace(x_min, x_max, x_split)
        y = np.linspace(y_min, y_max, y_split)
        X, Y = np.meshgrid(x, y)
        F = function(X, Y)

        fig, ax = plt.subplots()
        plt.grid(True)
        for i in range(0, len(path[0]) - 1):
            ax.contour(X, Y, F - function(path[0][i], path[1][i]), levels=[0])
            plt.plot([path[0][i], path[0][i + 1]], [path[1][i], path[1][i + 1]], c='r')
        ax.contour(X, Y, F - function(path[-1][0], path[-1][1]), levels=[0])
        ax.plot(a_x, a_y, 'ro')
        plt.xlabel("x")
        plt.ylabel("y")
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)

        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(a_x, a_y, function(a_x, a_y), color='red')
        ax.plot_surface(X, Y, F, rstride=5, cstride=5, alpha=0.7)
        for i in range(0, len(path[0]) - 1):
            ax.plot([path[0][i], path[0][i + 1]], [path[1][i], path[1][i + 1]],
                    [function(path[0][i], path[1][i]),
                     function(path[0][i + 1], path[1][i + 1])], c='r')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)

        fig, ax = plt.subplots()
        error = []
        plt.grid(True)
        for i in range(0, len(path[0])):
            error.append(abs(function(a_x, a_y) - function(path[0][i], path[1][i])))
        ax.plot(range(1, len(error) + 1), error)
        plt.xlabel("iteration")
        plt.ylabel("error")
        name = "image/image" + str(i) + "_" + str(time.time()) + ".png"
        plt.savefig("web/" + name, transparent=True)
        answer[-1].append(name)
        plt.close('all')
        answer[-1].append(k)
    return answer


eel.init('web')

eel.start('index.html')

