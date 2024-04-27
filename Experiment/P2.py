import matplotlib.pyplot as plt
import numpy as np

beta = lambda zeta, s: np.sqrt(1 / ((1 - s ** 2) ** 2 + (2 * zeta * s) ** 2))
theta = lambda zeta, s: np.arctan(2 * zeta * s / (1 - s ** 2))
zetas = np.array([0, 0.1, 0.25, 0.375, 0.5, 1])
s = np.arange(300) / 100

fig, axs = plt.subplots(1, 2)
for zeta in zetas:
    axs[0].plot(s, beta(zeta, s), label=f'zeta={zeta:.2f}')
    y = np.degrees(theta(zeta, s))
    for idx, elem in enumerate(y):
        if idx > int(len(y) / 3):
            y[idx] = 180 + elem
    axs[1].plot(s, y, label=f'zeta={zeta:.2f}')

axs[0].set_ylim(0, 6)
axs[1].set_ylim(-0.5, 180.5)
axs[0].set_xlabel('s')
axs[1].set_xlabel('s')
axs[0].set_ylabel('Beta')
axs[1].set_ylabel('Theta')
axs[0].set_title('Beta(s)')
axs[1].set_title('Theta(s)')
axs[0].legend()
axs[1].legend()
plt.show()
