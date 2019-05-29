# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""
import numpy as np
import os
from simulated_annealing import simulated_annealing
import matplotlib.pyplot as plt


def compute_tour_length(perm, graph):
    t_len = 0.0
    for k in range(len(perm)):
        t_len += graph[perm[k], perm[np.mod(k+1, len(perm))]]
    return t_len


def insert(xx):
    x = np.copy(xx)
    n = len(x)
    move = np.random.randint(0, n-1)
    index = np.random.randint(0, n-1)
    temp = np.delete(x, move)
    to_ret = np.insert(temp, index, x[move])
    return to_ret


def swap(xx):
    x = np.copy(xx)
    num_swaps = np.random.randint(1, 3)
    for _ in range(np.int(num_swaps)):
        p = np.random.randint(0, len(xx) - 1)
        p2 = np.mod((p+1), len(xx))
        tmp = x[p]
        x[p] = x[p2]
        x[p2] = tmp
    return x


def inverse(xx):
    arr = xx.tolist()
    p1 = np.random.randint(0, len(arr) - 1)
    p2 = np.random.randint(0, len(arr) - 1)
    a = min(p1, p2)
    b = max(p1, p2)
    to_ret = arr[:a] + arr[a:b][::-1] + arr[b:]
    return np.array(to_ret)


def variation(xx, func, graph):
    x1 = swap(xx)
    x2 = inverse(xx)

    f1 = func(x1, graph)
    f2 = func(x2, graph)

    fmin = min(f1, f2)
    if fmin == f1:
        return x1, f1
    else:
        return x2, f2


if __name__ == "__main__":
    dirname = ""
    fname = os.path.join(dirname, "hachula130.dat")
    data = []
    NTrials = (10**6)/2
    with open(fname) as f:
        for line in f:
            data.append(line.split())
    n = len(data)
    G = np.empty([n, n])
    for i in range(n):
        for j in range(i, n):
            G[i, j] = np.linalg.norm(np.array([float(data[i][1]), float(data[i][2])]) - np.array([float(data[j][1]),
                                                                                                  float(data[j][2])]))
            G[j, i] = G[i, j]

    xbest, fbest, history = simulated_annealing(n, NTrials, G, variation, compute_tour_length)
    print("fbest ", fbest, "xbest ", xbest)
    plt.plot(history)
    plt.show()