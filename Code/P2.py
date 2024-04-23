import matplotlib.pyplot as plt
import numpy as np

beta = lambda zeta, s: np.sqrt(1 / ((1 - s ** 2) ** 2 + (2 * zeta * s) ** 2))
theta = lambda zeta, s: np.arctan(2 * zeta * s / (1 - s ** 2))
zetas = np.array([0, 0.1, 0.25, 0.375, 0.5, 1])
s = np.arange(300) / 100

fig, axs = plt.subplots(1, 2)
for zeta in zetas:
    axs[0].plot(s, beta(zeta, s))
    y = np.degrees(theta(zeta, s))
    for idx, elem in enumerate(y):
        if idx > int(len(y) / 3):
            y[idx] = 180 + elem
    axs[1].plot(s, y)

axs[0].set_ylim(0, 6)
axs[0].set_title('Beta(s)')
axs[1].set_title('Theta(s)')
plt.show()
