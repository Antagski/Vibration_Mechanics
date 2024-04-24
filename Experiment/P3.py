import numpy as np
import matplotlib.pyplot as plt


def main():
    square_wave = lambda i, t: 4 * np.sin((2 * i - 1) * t) / (np.pi * (2 * i - 1))
    sawtooth_wave = lambda i, t: (-1) ** i * 2 * np.sin(i * t) / (np.pi * i)

    t = np.linspace(0, 2 * np.pi, 100)
    plt.ioff()
    fig, axs = plt.subplots(2, 1)
    wave = np.zeros((2, len(t)))
    for i in range(10):
        n = i + 1
        w_1 = square_wave(n, t)
        w_2 = sawtooth_wave(n, t)
        wave[0, :] = wave[0, :] + w_1[:]
        wave[1, :] = wave[1, :] + w_2[:]
        axs[0].plot(t, wave[0, :])
        axs[1].plot(t, wave[1, :])
    plt.show()



if __name__ == "__main__":
    main()
