import numpy as np
from numpy import single
from numpy.typing import NDArray as Arr
import matplotlib.pyplot as plt


def gauss(A: Arr[Arr[single]], b: Arr[single], pivoting: int=0):
    n = len(A)
    for i in range(n - 1):
        if pivoting:
            mx = -1
            ind = i
            for j in range(i, n):
                if abs(A[j][i]) > mx:
                    ind = j
                    mx = abs(A[j][i])
            A[[ind, i]] = A[[i, ind]]
            b[[ind, i]] = b[[i, ind]]
        for j in range(i + 1, n):
            m = A[j][i] / A[i][i]
            b[j] -= b[i] * m
            for k in range(i, n):
                A[j][k] -= A[i][k] * m
    x = np.zeros(n, dtype=np.float32)
    x[-1] = b[-1] / A[-1][-1]
    for i in range(n - 2, -1, -1):
        x[i] = b[i]
        for j in range(n - 1, i, -1):
            x[i] -= x[j] * A[i][j]
        x[i] = x[i] / A[i][i]
    return x


def thomas(A: Arr[Arr[single]], b: Arr[single]):
    n = len(A)
    gamma = np.zeros(n)
    beta = np.zeros(n)
    for i in range(n - 1):
        gamma[i + 1] = -(A[i][i + 1] / (A[i][i - 1] * gamma[i] + A[i][i]))
        beta[i + 1] = (b[i] - A[i][i - 1] * beta[i]) / (A[i][i - 1] * gamma[i] + A[i][i])
    x = np.zeros(n)
    x[-1] = (b[-1] - A[-1][-2] * beta[-1]) / (A[-1][-1] + A[-1][-2] * gamma[-1])
    for i in range(n - 1, 0, -1):
        x[i - 1] = gamma[i] * x[i] + beta[i]
    return x


def generator(n: int, three: bool=False):
    while True:
        m = 2 * np.random.random_sample((n, n)) - 1
        m = np.array(m, dtype=np.float32)
        if three:
            for i in range(6):
                for j in range(6):
                    if abs(j - i) > 1:
                        m[i][j] = 0
        if np.linalg.det != 0:
            return m


a = [np.array([2., 4., 3., 2.]), np.array([6., 6., 8., 11.]),
     np.array([-2., 5., 8., 3.]), np.array([9., 3., 1., 10.])]
a = np.array(a)
b = np.array([3., 8., 2., 4.])
print(gauss(a, b, 1))

count = 1000

a = np.array([generator(6) for i in range(count)])
a3 = np.array([generator(6, True) for i in range(count)])

r_1 = [np.linalg.eigvals(a[i]) for i in range(count)]
plt.hist([max(i) for i in r_1])
plt.grid()
plt.xlabel("Спектральный радиус")
plt.ylabel("Количество матриц")
plt.show()
print([max(np.abs(i)) / min(np.abs(i)) for i in r_1])

r_2 = [np.linalg.eigvals(a3[i]) for i in range(count)]
print([max(np.abs(i)) / min(np.abs(i)) for i in r_2])
plt.hist([max(i) for i in r_2])
plt.grid()
plt.xlabel("Спектральный радиус")
plt.ylabel("Количество матриц")
plt.show()
bins = [i for i in range(0, 100, 10)]
c_1 = [np.linalg.cond(a[i]) for i in range(count)]
plt.hist(c_1, bins=bins)
plt.grid()
plt.xlabel("Число обусловленности")
plt.ylabel("Количество матриц")
plt.show()
c_2 = [np.linalg.cond(a3[i]) for i in range(count)]
plt.hist(c_2, bins=bins)
plt.grid()
plt.xlabel("Число обусловленности")
plt.ylabel("Количество матриц")
plt.show()



ans_a = np.array([gauss(a[i], np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32)) for i in range(count)])
a_acc = np.array([gauss(a[i], np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32), 1) for i in range(count)])

ans_a3 = np.array([thomas(a3[i], np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32)) for i in range(count)])
diag_acc = np.array([gauss(a3[i], np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32), 1)
                            for i in range(count)])

square = np.array([np.sqrt(sum(np.array([np.square(-ans_a[i][j] + a_acc[i][j]) for j in range(6)]))) /
                   np.sqrt(sum(np.array([np.square(a_acc[i][j]) for j in range(6)])))
                   for i in range(count)], dtype=np.float32)

square_diag = np.array([np.sqrt(sum(np.array([np.square(- ans_a3[i][j] + diag_acc[i][j]) for j in range(6)]))) /
                    np.sqrt(sum(np.array([np.square(diag_acc[i][j]) for j in range(6)])))
                   for i in range(count)], dtype=np.float32)

sup = np.array([np.max(np.array([np.abs(- ans_a[i][j] + a_acc[i][j]) for j in range(6)])) /
                np.max(np.array([np.abs(a_acc[i][j]) for j in range(6)]))
                for i in range(count)])

sup_diag = np.array([np.max(np.array([np.abs(- ans_a3[i][j] + diag_acc[i][j]) for j in range(6)])) /
                np.max(np.array([np.abs(diag_acc[i][j]) for j in range(6)]))
                for i in range(count)])

bins = [0.4 * i for i in range(10)]
plt.hist(sup, bins=bins)
plt.grid()
plt.xlabel("Относительная погрешность")
plt.ylabel("Количество матриц")
plt.show()

bins = [10**(-9) * i for i in range(0, 1000, 100)]
plt.hist(sup_diag, bins=bins)
plt.xlabel("Относительная погрешность")
plt.ylabel("Количество матриц")
plt.grid()
plt.show()

print(square)
bins = [0.4 * i for i in range(10)]
plt.grid()
plt.xlabel("Относительная погрешность")
plt.ylabel("Количество матриц")
plt.hist(square, bins=bins)
plt.show()

bins = [10**(-9) * i for i in range(0, 1000, 100)]
plt.hist(square_diag, bins=bins)
plt.grid()
plt.xlabel("Относительная погрешность")
plt.ylabel("Количество матриц")
plt.show()


def generator_positive(n: int):
    while True:
        m = np.array([np.zeros(n) for i in range(n)], dtype=np.float32)
        mx = 0.
        for i in range(n):
            m[i][i] = np.random.random()
            if m[i][i] > mx:
                mx = m[i][i]
        for i in range(n):
            for j in range(i + 1):
                if i != j:
                    val = m[i][i] * m[j][j]
                    if i % 2 == 0:
                        if j % 2 == 1:
                            zn = -1
                        else:
                            zn = 1
                    else:
                        if j % 2 == 1:
                            zn = 1
                        else:
                            zn = -1
                    m[i][j] = np.sqrt(val * np.random.random())
                    while m[i][j] >= mx:
                        m[i][j] = np.sqrt(val * np.random.random())
                    m[i][j] = m[i][j] * zn
                    m[j][i] = m[i][j]
        flag = 1
        for i in range(1, n):
            minor = np.array([row[:i+1] for row in m[:i+1]])
            if np.linalg.det(minor) < 0:
                flag = 0
                break
        if flag:
            return m


def cholesky(A: Arr[Arr[single]], b: Arr[single]):
    n = len(A)
    l = np.array([np.zeros(n) for i in range(n)], dtype=np.float32)
    lt = np.array([np.zeros(n) for i in range(n)], dtype=np.float32)
    for i in range(n):
        for j in range(i):
            l[i][j] = (1 / l[j][j]) * (A[i][j] - np.sum(np.array([l[i][k] * l[j][k] for k in range(j)])))
            lt[j][i] = l[i][j]
        l[i][i] = np.sqrt(A[i][i] - np.sum(np.array([l[i][j] ** 2 for j in range(i)])))
        lt[i][i] = l[i][i]
    # print(np.matmul(l, lt))
    y = np.zeros(n, dtype=np.float32)
    y[0] = b[0] / l[0][0]
    for i in range(1, n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= y[j] * l[i][j]
        y[i] = y[i] / l[i][i]
    x = np.zeros(n, dtype=np.float32)
    x[-1] = y[-1] / lt[-1][-1]
    for i in range(n - 2, -1, -1):
        x[i] = y[i]
        for j in range(n - 1, i, -1):
            x[i] -= x[j] * lt[i][j]
        x[i] = x[i] / lt[i][i]
    return x


a = np.array([generator_positive(6) for i in range(count)], dtype=np.float32)
ans_ch = np.array([cholesky(a[i], np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32)) for i in range(count)])
ch_acc = np.array([gauss(a[i], np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32), 1) for i in range(count)])
s_ch = np.array([np.sqrt(sum(np.array([np.square(- ans_ch[i][j] + ch_acc[i][j]) for j in range(6)]))) /
                      np.sqrt(sum(np.array([np.square(ch_acc[i][j]) for j in range(6)])))
                      for i in range(count)], dtype=np.float32)

sup_ch = np.array([np.max(np.array([np.abs(- ans_ch[i][j] + ch_acc[i][j]) for j in range(6)])) /
                   np.max(np.array([np.abs(ch_acc[i][j]) for j in range(6)]))
                   for i in range(count)])
print(s_ch)

bins = [10**(-8) * i for i in range(0, 1000, 100)]
plt.hist(sup_ch, bins=bins)
plt.grid()
plt.xlabel("Относительная погрешность")
plt.ylabel("Количество матриц")
plt.show()

bins = [10**(-8) * i for i in range(0, 1000, 100)]
plt.hist(s_ch, bins=bins)
plt.grid()
plt.xlabel("Относительная погрешность")
plt.ylabel("Количество матриц")
plt.show()

r_3 = [np.linalg.eigvals(a[i]) for i in range(count)]
plt.hist([max(i) for i in r_3])
plt.grid()
plt.xlabel("Спектральный радиус")
plt.ylabel("Количество матриц")
plt.show()
bins = [i for i in range(0, 100, 10)]
c_3 = [np.linalg.cond(a[i]) for i in range(count)]
plt.hist(c_1, bins=bins)
plt.grid()
plt.xlabel("Число обусловленности")
plt.ylabel("Количество матриц")
plt.show()