from typing import Callable, Union, List, Optional

import numpy as np
from numpy import single, array
from numpy.typing import NDArray as Arr
from matplotlib import pyplot as plt
from scipy.optimize import fsolve


def functional(x: single):
    c = single(1.03439984)
    if x == single(0):
        return 0
    t_x = fsolve(lambda t: c * t - c * 0.5 * np.sin(2*t) - x, np.array([1.75418438]))
    y_x = c * (0.5 - 0.5 * np.cos(2 * t_x))
    y_dx = np.sin(2 * t_x) / (1 - np.cos(2 * t_x))
    return np.sqrt((1 + (y_dx**2)) / (2 * 9.81 * y_x))


# Пункт 1
def composite_simpson(a: float, b: float, n: int, func: Callable) -> single:
    if n % 2 == 0:
        n += 1
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    f = []
    for i in x:
        f.append(func(i))
    res = (h / 3. * (f[0] +
                     2 * np.sum([f[i] for i in range(2, len(x) - 1, 2)]) +
                     4 * np.sum([f[i] for i in range(1, len(x), 2)])
                     + f[-1])
           )
    return res


def composite_trapezoid(a: float, b: float, n: int, func: Callable) -> single:
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    f = []
    for i in x:
        f.append(func(i))
    res = (h / 2. * (f[0] + 2 * np.sum([f[i] for i in range(2, len(x) - 1)]) + f[-1]))
    return res


# Пункт 2
def num_integration(func: Callable, a: float,
                    b: float, form: Callable, n: List[int]) -> Arr[single]:
    res = []
    for i in n:
        res.append(form(a, b, i, func))
    return np.array(res)


def regression_cf(x: Arr[single], y: Arr[single]) -> Arr[single]:
    n = len(x)
    a0 = (np.sum(x ** 2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / (
            n * np.sum(x ** 2) - np.sum(x) ** 2
    )
    a1 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x ** 2) - np.sum(x) ** 2
    )
    return np.array([a0, a1])


# Пункт 3, пункт 4
def calculate(a: float, b: float) -> None:
    n = np.logspace(np.log10(3), np.log10(9999), 100, dtype=int)
    h = (b - a) / (n - 1)
    exact = np.sqrt((2 * 1.03439984) / 9.81) * 1.75418438

    res_simp = num_integration(functional, a, b, composite_simpson, n)
    np.savetxt('res_simp.txt', res_simp)
    res_trap = num_integration(functional, a, b, composite_trapezoid, n)
    np.savetxt('res_trap.txt', res_trap)

    res_simp = np.loadtxt('res_simp.txt')
    res_trap = np.loadtxt('res_trap.txt')
    error_simp = np.array([abs(exact - i) for i in res_simp])
    error_trap = np.array([abs(exact - i) for i in res_trap])

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax1.loglog(h, error_simp, 'o', label='Отклонения для формулы Симпсона')
    ax1.loglog(h, error_trap, 'o', label='Отклонение для формулы трапеций')
    ax1.set_xlabel('h')
    ax1.set_ylabel('E')
    lgs = np.logspace(-3, -1, 100)
    a0, a1 = regression_cf(h, error_simp)
    ax1.loglog(lgs, [i ** a1 for i in lgs], label="h**(a1) для формулы Симпсона")
    a0, a1 = regression_cf(h, error_trap)
    ax1.loglog(lgs, [i ** a1 for i in lgs], label="h**(a1) для формулы трапеций")
    ax1.loglog(lgs, [i ** 2 for i in lgs], label="h**2")
    ax1.loglog(lgs, [i ** 4 for i in lgs], label="h**4")
    ax1.grid()
    ax1.legend()

    plt.show()


if __name__ == '__main__':
    calculate(0., 2.)
