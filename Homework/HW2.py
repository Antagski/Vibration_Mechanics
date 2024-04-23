import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


GRAVITY = np.array([9.8])
LENGTH = np.array([1])
mass_1 = np.arange(0, 1, step=0.01)
mass = np.array([1])
rho = mass_1 / LENGTH
omega = [np.sqrt((r * GRAVITY * LENGTH / 2 + mass * GRAVITY) / (r * LENGTH**2 / 3 + mass * LENGTH)) for r in rho]
plt.plot(rho, omega)
plt.xlabel("Mass of bar")
plt.ylabel("Omega")
plt.title("Omega vs Mass of bar")
plt.legend(["Mass of bar"])
plt.grid(True)
plt.show()
