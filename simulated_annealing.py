# -*- coding: utf-8 -*-
"""
@author: morteltsur@gmail.com
"""
import numpy as np


def simulated_annealing(n, max_evals, graph, variation, func=lambda x: x.dot(x), seed=None):
    T_init = 300
    T_min = 1e-4
    f_lower_bound = 0
    eps_satisfactory = 1e-5
    max_internal_runs = 100
    local_state = np.random.RandomState(seed)
    alpha = 0.9985
    history = []
    xbest = xmin = local_state.permutation(n)
    fbest = fmin = func(xmin, graph)
    eval_cntr = 1
    T = T_init
    history.append(fmin)
    while T > T_min and eval_cntr < max_evals:
        for i in range(max_internal_runs):
            x, f_x = np.copy(variation(xmin, func, graph))
            eval_cntr += 1
            dE = f_x - fmin
            check = local_state.uniform()
            if dE <= 0:
                xmin = x
                fmin = f_x
            elif check < np.exp(-dE/T):
                xmin = x
                fmin = f_x
            if fmin < fbest:
                fbest = fmin
                xbest = xmin
            history.append(fmin)
            if np.mod(eval_cntr, int(max_evals/100)) == 0:
                print("T: ", T, "Iter: ", eval_cntr, " evals: fmin=", fmin)
            if fbest < f_lower_bound+eps_satisfactory:
                T = T_min
                break
        T *= alpha
    return xbest, fbest, history