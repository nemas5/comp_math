import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
import timeit

def calculate(t, x):
    ans = np.array([x[1], np.cos(t) - 0.1*x[1] - np.sin(x[0])])
    return ans

def runge_kutta(x_0, t_n, f, h):
    x = np.zeros((int(t_n / h) + 1, 2))
    x[0] = [0, x_0]
    for i in range(int(t_n / h)):
        k1 = h * f(i * h, x[i])
        k2 = h * f(i * h + h / 2, x[i] + 0.5 * k1)
        k3 = h * f(i * h + h / 2, x[i] + 0.5 * k2)
        k4 = h * f(i * h + h, x[i] + k3)
        x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x

def root(x, t, x_pre, f, h):
    return x - x_pre[1] - h * ((2/3)*f(t[1], x_pre[1]) - (1/12)*f(t[0], x_pre[0]) + (5/12)*f(t[2], x))

def adams_moulton(x_0, t_n, f, h):
    x = np.zeros((int(t_n / h) + 1, 2))
    x[:2] = runge_kutta(x_0, t_n, f, h)[:2]
    for i in range(1, int(t_n / h)):
        x[i + 1] = scipy.optimize.root(root, x[i],
                                       args=([h*(i - 1), h*i, h*(i+1)], [x[i - 1], x[i]], f, h)).x
    return x

def milne_simpson(x_0, t_n, f, h):
    x = np.zeros((int(t_n / h) + 1, 2))
    x[:4] = runge_kutta(x_0, t_n, f, h)[:4]
    for i in range(3, int(t_n / h)):
        x[i + 1] = x[i - 3] + ((4*h/3) * (2 * f(i * h, x[i]) - f(h*(i - 1), x[i - 1]) + 2 * f(h*(i - 2), x[i - 2])))
        x[i + 1] = x[i - 1] + ((h/3) * (f(h*(i + 1), x[i + 1]) + 4 * f(h*i, x[i]) + f(h*(i - 1), x[i - 1])))
    return x

def plot_trajectory(method, t_n, f, h):
    for i in np.linspace(1.85, 2.1, 15):
        res = method(i, t_n, f, h)
        plt.plot([i * h for i in range(int(t_n / h) + 1)], res[:, 0], label = 'x0 =' + str(round(i, 4)))
    plt.xlabel('t')
    plt.ylabel('θ')
    plt.legend()
    plt.show()

t = 200
h = 0.1
plot_trajectory(runge_kutta, t, calculate, h)
plot_trajectory(adams_moulton, t, calculate, h)
plot_trajectory(milne_simpson, t, calculate, h)

def find_h(method, t_n, f, x_0):
    for i in np.linspace(0.1, 2, 10):
        res = method(x_0, t_n, f, i)
        plt.plot([j * i for j in range(int(t_n / i) + 1)], res[:, 0], label='h =' + str(round(i, 3)))
    plt.xlabel('t')
    plt.ylabel('θ')
    plt.legend()
    plt.show()

find_h(runge_kutta, 1000, calculate, 1.85)
find_h(adams_moulton, 1000, calculate, 1.85)
find_h(milne_simpson, 100, calculate, 1.85)

def phase(method, t_n, f, h):
    for i in np.linspace(1.85, 2.1, 15):
        res = method(i, t_n, f, h)
        plt.plot(res[:, 0], res[:, 1], label = 'x0 =' + str(round(i, 4)))
    plt.xlabel('θ')
    plt.ylabel("θ'")
    plt.legend()
    plt.show()

phase(runge_kutta, t, calculate, h)
phase(adams_moulton, t, calculate, h)
phase(milne_simpson, t, calculate, h)


def time_measure(method, t_n, f, h, x_0, name):
    t0 = timeit.default_timer()
    res = method(x_0, t_n, f, h)
    t1 = timeit.default_timer()
    plt.plot(res[:, 0], res[:, 1], label=name)
    return t1 - t0

t1 = time_measure(runge_kutta, 200, calculate, 0.1, 1.85, 'Рунге-Кутта')
t2 = time_measure(adams_moulton, 200, calculate, 0.1, 1.85, 'Адамсон-Моултон')
t3 = time_measure(milne_simpson, 200, calculate, 0.1, 1.85, 'Милн-Симпсон')
plt.xlabel('θ')
plt.ylabel("θ'")
plt.legend()
plt.show()
print(t1, t2 - t1, t3 - t1)
