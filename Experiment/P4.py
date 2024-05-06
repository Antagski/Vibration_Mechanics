import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    square_wave = lambda i, t: 4 * np.sin((2 * i - 1) * t) / (np.pi * (2 * i - 1))
    sawtooth_wave = lambda i, t: (-1) ** i * 2 * np.sin(i * t) / (np.pi * i)

    t = np.linspace(0, 2 * np.pi, 100)
    plt.ioff()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    z = 0
    wave = np.zeros((2, len(t)))
    for i in range(10):
        n = i + 1
        w_1 = square_wave(n, t)
        w_2 = sawtooth_wave(n, t)
        z += 1
        wave[0, :] = wave[0, :] + w_1[:]
        wave[1, :] = wave[1, :] + w_2[:]
        ax1.plot(t, n*np.ones_like(t), w_1)
        ax2.plot(t, n*np.ones_like(t), w_2)
    ax1.plot(t, (n + 1) * np.ones_like(t), wave[0, :])
    ax2.plot(t, (n + 1) * np.ones_like(t), wave[1, :])
    ax1.set_box_aspect((3, 3, 2))
    ax2.set_box_aspect((3, 3, 2))
    ax1.set_xlabel("Time duration")
    ax1.set_ylabel("Frequency view")
    ax1.set_zlabel("Amplitude/Magnitude")
    ax2.set_xlabel("Time duration")
    ax2.set_ylabel("Frequency view")
    ax2.set_zlabel("Amplitude/Magnitude")

    plt.show()

if __name__ == "__main__":
    main()
