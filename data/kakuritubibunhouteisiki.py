import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sigma = 2
mu = 1
Xzero = 1

T = 1.0
N = 2 ** 8
dt = T / N

t = np.arange(0.0, T, dt)
dW = np.sqrt(dt) * np.random.randn(N)
dW[0] = 0
W = np.cumsum(dW)
Xtrue = Xzero * np.exp((sigma - 0.5 * mu ** 2) * t + (mu * W))

Xem = np.zeros(N)
Xtemp = Xzero
Xem[0] = Xtemp
for j in range(1, N):
    Xtemp = Xtemp + dt * sigma * Xtemp + mu * Xtemp * dW[j]
    Xem[j] = Xtemp

plt.plot(t, Xtrue, label='Analytic')
plt.plot(t, Xem, label='Euler-Maruyama')
plt.legend()
plt.show()
