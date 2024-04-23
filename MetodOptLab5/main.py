import numpy as np
from logic import simplex
from logic.GoldenSectionMethod import *
import matplotlib.pyplot as plt


# bazaara book
# the code solves problems of the type as given on page 550
# following solves the example 10.1.8 as given on page 551

def f_func(X):
    x1 = X[0]
    x2 = X[1]
    return 6*(x1**2) + (x2**2) - 2*x1*x2 - 10*x2

def fZ(x1,x2):
    return 6*(x1**2) + (x2**2) - 2*x1*x2 - 10*x2

def g_list(X):
    x1 = X[0]
    x2 = X[1]
    return np.array([2*x1 + x2 - 5, -2 * x1 - x2 + 2, -1*x1, -1*x2])


def find_active_set(X):
    return [i for (i,c) in enumerate(X) if abs(c) < 1e-3 ]

def grad_f(X):
    eps = 1e-6
    diff = np.zeros((X.size, X.size))
    for i in range(X.size):
        diff[i][i]=eps
    return np.resize(np.array([(f_func(X+diff[i])-f_func(X-diff[i]))/(2*eps) for i in range(X.size)]), (X.size,1))

def grad_g(X):
    eps = 1e-6
    diff=np.zeros((X.size,X.size))
    for i in range(X.size):
        diff[i][i]=eps
        n=g_list(X).size
    return np.array([[(g_list(X+diff[i])[j]-g_list(X-diff[i])[j])/(2*eps) for i in range(X.size)] for j in range(n)])


X_list = np.zeros((1,2))
X0 = np.array((0, 4))
X_list[0] = X0
z = -24
i = 0
n = X_list[0].size

while True:
    current=X_list[i]
    print('X:', current)
    print('g_i:', g_list(current))
    active_set=find_active_set(g_list(current))
    f_grad=grad_f(current)
    g_grad=grad_g(current)
    a11=-1*f_grad
    g_i=np.array([g_grad[a] for a in active_set])
    g_i=g_i.T
    a12=-1*g_i
    if(len(active_set)>0):
        A1=np.concatenate((a11, a12), axis=1)
    else:
        A1=a11
    a21=f_grad
    a22=g_i
    if(len(active_set)>0):
        A2=np.concatenate((a21, a22), axis=1)
    else:
        A2=a21
    A1=np.concatenate((A1, np.eye(n), -1*np.eye(n)), axis=1)
    A2=np.concatenate((A2, -1*np.eye(n), np.eye(n)), axis=1)
    A3=np.ones((1, (1+len(active_set))))
    A3=np.concatenate((A3, np.zeros((1, 2*n))), axis=1)
    A4=-1*A3
    A=np.concatenate((A1, A2, A3, A4), axis=0)
    c=np.concatenate((np.zeros((1,1+len(active_set))), np.ones((1,2*n))), axis=1)
    final_matrix_1=np.concatenate((A,c), axis=0)
    final_matrix_2=np.zeros((2*n+3, 1))
    final_matrix_2[-2]=-1
    final_matrix_2[-3] = 1
    A=np.concatenate((final_matrix_1, final_matrix_2), axis=1)

    x, y, z = simplex.solve(A, n)
    d = y[0:n] - y[n:2*n]

    print('X:', X_list[-1])
    print('active_set', active_set)
    print(A)
    
    print('d', d, 'z', z)
    if (abs(z)<=1e-3):
        break

    print("lambda")
    lambda_max = 1
    while(lambda_max>=0):
        temp=g_list(current+lambda_max*d)
        if (np.sum(temp>0)==0):
            break
        lambda_max -= 1e-5
    print('lambda_max', lambda_max)
    
    # new_func=lambda x : f_func(X_list[i]+x*d)
    # x_ = np.arange(-1, 1, 0.001)
    # y_ = np.array([new_func(j) for j in x_])
    # import matplotlib.pyplot as plt; plt.plot(x_, y_); plt.show();
    
    lambda_ans = goldenSectionSearch(lambda x : f_func(X_list[i]+x*d),0, lambda_max , 1e-5)
    print('lambda_ans', lambda_ans)
    print("lambda_end")

    
    X_list=np.concatenate((X_list, np.resize(current+lambda_ans*d, (1, n))), axis=0)
    i+=1
    print("i: " + str(i))

# list of points
print(X_list)

x = np.linspace(0, 0.75, 100)
y = np.linspace(3, 5, 100)
X, Y = np.meshgrid(x, y)
F = fZ(X, Y)

ax = plt.figure().add_subplot(111, projection='3d')
ax.plot_surface(X, Y, F, rstride=5, cstride=5, alpha=0.7)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

fig, ax = plt.subplots()
plt.grid(True)
for i in range(0, 5):
    ax.contour(X, Y, F + 30 - i, levels=[0])
#print(grad_g(X0).shape)
#print(grad_f(X0))


# plt.figure()
# plt.subplot(121, aspect='equal')
# plt.axes()
# delta = 0.1
# x = np.arange(-5, 5, delta)
# y = np.arange(-5, 5, delta)
# X, Y = np.meshgrid(x, y)
#
# Z = fZ(X,Y)

g1 = lambda x1,x2: 2*x1 + x2 - 5
g1 = g1(X,Y)
plt.plot(X_list[:,0], X_list[:,1], 'r')
plt.contour(X, Y, g1, colors=('b'), linewidths=1, levels=[0])

g1 = lambda x1,x2: -2 * x1 - x2 + 2
g1 = g1(X,Y)
plt.contour(X, Y, g1, colors=('g'), linewidths=1, levels=[0])
# plt.plot(X_list[:,0], X_list[:,1], 'r')

plt.savefig('test3.png', dpi=600)
plt.show()
