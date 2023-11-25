import math
from typing import Union, List

import numpy as np
from numpy import single, array
from numpy.typing import NDArray as Arr
import matplotlib.pyplot as plt


plt.xlabel('x')
plt.ylabel('y')


def plot_contour(file: str) -> Arr[Arr[single]]:  # Пункт 2, визуализация контура
    points = np.loadtxt(file)
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y, s=1, label='Множество точек P')
    plt.legend()
    return points


def build_a_matrix(n: int, h: float) -> Arr[Arr[single]]:  # Пункт 4
    a = array([array([0. for i in range(n)]) for j in range(n)])
    a[0][0] = 1.
    a[-1][-1] = 1.
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if i == j:
                a[i][j] = 4 * h
                a[i][j - 1] = h
                a[i][j + 1] = h
                continue
    a = np.linalg.inv(a)
    return a


def build_b_matrix(pts: Arr[single], n: int) -> Arr[single]:  # Пункт 4
    b = array([0. for i in range(n)])
    for i in range(1, n - 1):
        b[i] = 3*(-2*pts[i] + pts[i - 1] + pts[i + 1])
    return b


def calculate_cfs(a: Arr[Arr[single]], b: Arr[single],
                  pts: Arr[Arr[single]], h: float, n: int) -> Arr[Arr[single]]:  # Пункт 4
    c = np.dot(a, b)
    a0 = pts[:-1]
    a1 = array([(1 / h) * (pts[i + 1] - pts[i]) - (h / 3) * (c[i + 1] + 2 * c[i]) for i in range(n - 1)])
    a2 = array([c[i] for i in range(n - 1)])
    a3 = array([(c[i + 1] - c[i]) / (3 * h) for i in range(n - 1)])
    return np.column_stack([a0, a1, a2, a3])


def calculate_pts(a: Arr[Arr[single]],
                  h: float, n: int, m: int, hj: float) -> list[single]:  # Пункт 6
    t = array([hj * j for j in range(m)])
    tj = array([h * j for j in range(n)])
    pts = list()
    for i in range(len(t)):
        for j in range(n - 1):
            if tj[j] <= t[i] < tj[j + 1]:
                pts.append(a[j][0] + a[j][1] * (t[i] - tj[j]) +
                           a[j][2] * (t[i] - tj[j])**2 + a[j][3] * (t[i] - tj[j])**3)
    return pts


def lab1_base(filename_in: str, factor: int, filename_out: str) -> list:  # Реализация базовой части
    # Пункт 2

    p = plot_contour(filename_in)
    plt.show()

    # Пункт 3
    p_sps = p[::factor]  # Задание разреженного множества узлов интерполяции
    x_sps = p_sps[:, 0]  # Разреженные множества x и y
    y_sps = p_sps[:, 1]
    n = len(p_sps)

    # Пункт 4
    h = 1.  # Шаг для задания последовательности tj
    a_inv = build_a_matrix(n, h)  # Построение инвертированной матрицы для составления разрешающих СЛАУ

    b_x = build_b_matrix(x_sps, n)
    b_y = build_b_matrix(y_sps, n)

    cfs_x = calculate_cfs(a_inv, b_x, x_sps, h, n)
    cfs_y = calculate_cfs(a_inv, b_y, y_sps, h, n)

    # Пункт 6
    plt.xlabel('x')
    plt.ylabel('y')
    hj = 0.1  # Шаг для t, подставляемых в полученные уравнения кубических сплайнов
    x = calculate_pts(cfs_x, h, n, len(p), hj)
    y = calculate_pts(cfs_y, h, n, len(p), hj)
    plot_contour(filename_in)
    plt.scatter(x, y, s=1, label='Точки кубического сплайна')
    plt.scatter(x_sps, y_sps, s=1, label='Узлы интерполяции')
    plt.legend()
    plt.show()

    # Пункт 5
    r = list()
    for i in range(len(x)):
        if i % 10 != 0:
            r.append(((x[i] - p[i, 0]) ** 2 + (y[i] - p[i, 1]) ** 2) ** 0.5)
    print("Среднее отклонение:", np.mean(r))
    print("Стандартное отклонение:", np.std(r))

    np.savetxt("dist.txt", r)

    # Пункт 7
    output = np.column_stack([cfs_x, cfs_y])
    np.savetxt(filename_out, output)

    return [cfs_x, cfs_y, x, y, h, hj, p]  # Функция возвращает данные, необходимые для продвинутой части


# Продвинутая часть
# Пункт 8
class AutoDiffNum:  # Класс, реализующий концепцию дуальных чисел
    def __init__(self, a: Union[single, int, float],
                 b: Union[single, int, float] = 1.):
        self.a = a
        self.b = b

    def __add__(self, other: Union["AutoDiffNum", int, float, single]):
        if type(other) != AutoDiffNum:
            return AutoDiffNum(self.a + other, self.b)
        return AutoDiffNum(self.a + other.a, self.b + other.b)

    def __mul__(self, other: Union["AutoDiffNum", int, float, single]):
        if type(other) != AutoDiffNum:
            return AutoDiffNum(self.a * other, self.b * other)
        return AutoDiffNum(self.a * other.a, self.b * other.a + self.a * other.b)


# Пункт 9
# Вычисление производной кубического сплайна в точке на основе дуальных чисел
def spline_auto_dif(t: float, a: List[single], tj: float) -> single:
    t = AutoDiffNum(t)
    t_del = t + (-tj)
    s_dif = a[0]
    s_dif = (t_del * a[1]) + s_dif
    s_dif = (t_del * t_del) * a[2] + s_dif
    s_dif = (t_del * t_del * t_del) * a[3] + s_dif
    return s_dif.b


def build_tan(x0: single, y0: single,
              x: single, k: single) -> List[single]:  # Пункт 9, построение касательной
    y = (x - x0) * k + y0
    dx = x - x0
    dy = y - y0
    r = (dx**2 + dy**2)**0.5
    dx = dx / r * 10**(-7)
    dy = dy / r * 10**(-7)
    if dy < 0 < dx:
        dy = -dy
        dx = -dx
    plt.arrow(x0, y0, dx, dy, width=5*10**(-9))
    return [x - x0, y - y0]


def build_norm(x0: single, y0: single, x1: single,
               y1: single, x: single, f: single) -> None:  # Пункт 10, построение нормали к вектору
    x2 = x - x0
    y2 = -(x1 * x2) / y1
    r = (x2**2 + y2**2)**0.5
    x2 = x2 / r * 10**(-7)
    y2 = y2 / r * 10**(-7)
    if (y0 + y2 * 1000 < f < y0 + y1) or (y0 + y1 < f < y0 + y2 * 1000):
        plt.arrow(x0, y0, x2, y2, width=5*10**(-9), color='red', label='Нормали')
    else:
        plt.arrow(x0, y0, -x2, -y2, width=5*10**(-9), color='red')


def plotting(cfs_x: Arr[single], cfs_y: Arr[single],
             x: Arr[single], y: Arr[single], h: float, hj: float):  # Пункт 11
    t = array([hj * j for j in range(len(x))])
    tj = array([h * j for j in range(len(cfs_x))])
    plt.scatter(x[400:600], y[400:600], s=1)
    for i in range(411, 600, 20):
        j = 0
        while t[i] > tj[j]:
            j += 1
        j -= 1
        a_x = [cfs_x[j][0], cfs_x[j][1], cfs_x[j][2], cfs_x[j][3]]
        s_x = spline_auto_dif(t[i], a_x, tj[j])
        a_y = [cfs_y[j][0], cfs_y[j][1], cfs_y[j][2], cfs_y[j][3]]
        s_y = spline_auto_dif(t[i], a_y, tj[j])

        k = s_y / s_x
        tan = build_tan(x[i], y[i], x[i + 5], k)
        build_norm(x[i], y[i], tan[0], tan[1], x[i + 5], y[i + 5])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')
    plt.show()
    plt.axis()


# Пункт 12
def dist(x1: single, x2: single, y1: single, y2: single) -> single:  # Функция вычисления расстояния между точками
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5


def optional(x: Arr[single], y: Arr[single],
             h: float, hj: float, p: Arr[Arr[single]], factor: int) -> None:
    t = [hj * i for i in range(len(x))]
    x_p = p[:, 0]
    y_p = p[:, 1]
    ans_x, ans_x0 = [], []
    ans_y, ans_y0 = [], []
    r = list()
    for i in range(len(t)):
        if i % factor == 0:
            continue
        left = (i // factor) * factor
        right = left + int(h / hj)
        if right == len(x):
            right -= 1
        while right - left > 2:
            mid = (left + right) // 2
            dist_left = dist(x_p[i], x[mid - 1], y_p[i], y[mid - 1])
            dist_right = dist(x_p[i], x[mid + 1], y_p[i], y[mid + 1])
            if dist_left < dist_right:
                right = mid + 1
            else:
                left = mid

        mid = (left + right) // 2
        dist_left = dist(x_p[i], x[left], y_p[i], y[left])
        dist_right = dist(x_p[i], x[right], y_p[i], y[right])
        dist_mid = dist(x_p[i], x[mid], y_p[i], y[mid])

        if dist_left <= dist_right and dist_left <= dist_mid:
            ans_x.append(x[left])
            ans_y.append(y[left])
            r.append(dist_left)
        elif dist_right <= dist_left and dist_right <= dist_mid:
            ans_x.append(x[right])
            ans_y.append(y[right])
            r.append(dist_right)
        else:
            ans_x.append(x[mid])
            ans_y.append(y[mid])
            r.append(dist_mid)

        ans_x0.append(x_p[i])
        ans_y0.append(y_p[i])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(ans_x0[::50], ans_y0[::50], s=1)
    plt.scatter(ans_x[::50], ans_y[::50], s=1)
    plt.show()
    r = array(r)
    np.savetxt('dist2.txt', r)

    mean = np.mean(r)
    print("\nПункт 12")
    print("Среднее отклонение:", mean)


if __name__ == '__main__':
    M = 10
    coeffs = lab1_base('contour.txt', M, 'coeffs.txt')
    plotting(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5])
    optional(coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6], M)
