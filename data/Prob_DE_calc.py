import random
import numpy as np
from numpy.random import *
from scipy.integrate import odeint

# geometric Brownian motion; GBM


def GBM(numICs, tSpan, seed, type="z"):  # function X = PendulumFn(x1range, x2range, numICs, tSpan, seed, max_potential)

    sigma = 2
    mu = 1
    Xzero = 1

    T = 1.0
    N = 11#2 ** 8
    dt = T / N

    t = tSpan#np.arange(0.0, T, dt)

    lenT = len(tSpan) - 1

    X = np.zeros(numICs * lenT)
    cnt = 1
    if type == "x" or type == "y":
        for j in range(numICs):
            np.random.seed(seed=j)
            dW = np.sqrt(dt) * np.random.randn(N)
            dW[0] = 0
            W = np.cumsum(dW)
            X_true = Xzero * np.exp((sigma - 0.5 * mu ** 2) * t + (mu * W))
            if type == "x":
                    X_temp = X_true[:-1]
            elif type == "y":
                    X_temp = X_true[1:]
            X[(cnt - 1) * lenT: lenT + (cnt - 1) * lenT] = X_temp
            cnt += 1
    else:
        for j in range(numICs):
            np.random.seed(seed=(numICs + j))
            dW = np.sqrt(dt) * np.random.randn(N)
            dW[0] = 0
            W = np.cumsum(dW)
            X_true = Xzero * np.exp((sigma - 0.5 * mu ** 2) * t + (mu * W))
            X_temp = X_true[:-1]
            X[(cnt - 1) * lenT: lenT + (cnt - 1) * lenT] = X_temp
            cnt += 1
    return X





    """delta, beta, alpha = 0.5, -1, 1
    def dynsys(x, t):
        dydt = np.zeros_like(x)
        dydt[0] = x[1]  # x[1, :]
        dydt[1] = -delta * x[1] - x[0] * (beta + alpha * x[0] ** 2)
        # print(dydt)
        return dydt

    lenT = len(tSpan)  # 11, 500

    X = np.zeros((numICs * lenT, 2))

    # randomly start from x1range(1) to x1range(2)
    # x1 = (x1range[1] - x1range[0]) * rand() + x1range[0]

    # randomly start from x2range(1) to x2range(2)
    # x2 = (x2range[1] - x2range[0]) * rand() + x2range[0]
    # x1 = uniform(-2, 2)

    # randomly start from x2range(1) to x2range(2)
    # x2 = uniform(-2, 2)

    if type == "y":
        lenT = len(tSpan) - 1
        count = 1
        for j in range(100 * numICs):  # j = 1:100*numICs
            x1 = uniform(x1range[0], x1range[1])
            x2 = uniform(x2range[0], x2range[1])
            ic = [x1, x2]
            temp = odeint(dynsys, ic, tSpan)
            # [T, temp] = odeint(dynsys, ic, tSpan)
            temp = temp[1:]

            X[(count - 1) * lenT: lenT + (count - 1) * lenT, :] = temp
            if count == numICs:
                break
            count = count + 1
        return X

    else:
        count = 1
        for j in range(100 * numICs):  # j = 1:100*numICs
            x1 = uniform(x1range[0], x1range[1])
            x2 = uniform(x2range[0], x2range[1])
            ic = [x1, x2]
            temp = odeint(dynsys, ic, tSpan)
            # [T, temp] = odeint(dynsys, ic, tSpan)

            X[(count - 1) * lenT: lenT + (count - 1) * lenT, :] = temp
            if count == numICs:
                break
            count = count + 1
        return X"""
