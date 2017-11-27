#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt

seed = 123

def main():
    d = 2
    N = 10
    mean = 5

    rng = np.random.RandomState(seed)
    x1 = rng.randn(N, d) + np.array([0, 0])
    x2 = rng.randn(N, d) + np.array([mean, mean])

    plt.scatter(x1[:, 0], x1[:, 1])
    plt.scatter(x2[:, 0], x2[:, 1])
    plt.show()

    x = np.concatenate((x1, x2), axis=0)
    print(x)
    _simple_perceptron(x, N, d)

def _simple_perceptron(x, N, d):
    w = np.zeros(d)
    b = 0
    while True:
        classified = True
        for i in range(N*2):
            delta_w = (t(i, N) - y(x[i], w, b))*x[i]
            delta_b = (t(i, N) - y(x[i], w, b))
            w += delta_w
            b += delta_b
            classified *= all(delta_w == 0) * (delta_b == 0)
        if classified:
            break
    print(w)
    print(b)
    print(y([0, 0], w, b))
    print(y([5, 5], w, b))

    plt.scatter(x[:N, 0], x[:N, 1])
    plt.scatter(x[N:, 0], x[N:, 1])
    _x = np.arange(-2, 6)
    plt.plot(_x, -(w[0]*_x+b)/w[1])
    plt.show()

def y(x, w, b):
    return step(np.dot(w, x)+b)


def step(x):
    return 1 * (x > 0)


def t(i, N):
    if i < N:
        return 0
    else:
        return 1

if __name__=='__main__':
    main()
