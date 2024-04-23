import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

k, m_2, l, D, c = 350e3, 250, 5, 1, 18708.28693
_s = lambda v, m: 2 * np.pi * v * np.sqrt((m_2 + m) / k) / l
_zeta = lambda m: c / (2 * np.sqrt(k * (m_2 + m)))
_theta_1 = lambda zeta, s: np.arctan(2 * zeta * s / (1 - s**2))
_theta_2 = lambda zeta, s: np.arctan(2 * zeta * s)
_theta = lambda theta_1, theta_2: theta_1 - theta_2
_beta_2 = lambda zeta, s: np.sqrt((1 + (2 * zeta * s)**2)
                                    / ((1 - s**2)**2 + (2 * zeta * s)**2))
_beta_2_expand = lambda m, v: np.sqrt((4*c**2*np.pi**2*v**2*l**2 + l**4*k**2)
                               / (l**4*k**2 + 16*np.pi**4*v**4*(m_2+m)**2 - 8*np.pi**2*v**2*l**2*(m_2+m)*k
                                  + 4*c**2*np.pi**2*v**2*l**2))

m = np.linspace(0, 750, 750)
v = np.linspace(0, 30, 300)

def plot_result(m, v):
    # 绘制不同m与v下的beta2
    plt.figure()
    for v_var in v:
        beta_2 = _beta_2(_zeta(m), _s(v_var, m))
        plt.plot(m, beta_2)
    plt.legend(v)
    plt.grid()
    plt.xlabel("Delta Mass")
    plt.ylabel("Beta_2")
    plt.show(block=True)


def _plot_2(m, v):
    # 仅作于验证课本beta2曲线
    plt.figure()
    zeta = _zeta(m)
    print(zeta)
    s = np.arange(0, 10, 0.1)
    for zeta_var in zeta:
        plt.plot(s, _beta_2(zeta_var, s))
    plt.legend(zeta)
    plt.show()


def _plot_3(m, v):
    # 用于验证plot_1
    plt.figure()
    for v_var in v:
        plt.plot(m, _beta_2_expand(m, v_var))
    plt.legend(v)
    plt.xlabel("Delta Mass")
    plt.ylabel("Beta_2")
    plt.show()


def plot_3D(m, v):
    # 绘制三维曲面
    m_mesh, v_mesh = np.meshgrid(m, v, indexing='ij')
    z_mesh = _beta_2_expand(m_mesh, v_mesh)
    # 找到 Z 轴最大值及其索引
    max_z = np.max(z_mesh)
    max_z_index = np.unravel_index(np.argmax(z_mesh), z_mesh.shape)
    max_z_x, max_z_y = m_mesh[max_z_index], v_mesh[max_z_index]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(m_mesh, v_mesh, z_mesh, cmap='hot',)
    bar = fig.colorbar(surf, shrink=0.5, aspect=5,)
    bar.set_label("Beta_2")

    ax.set_xlabel("Delta Mass")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Beta_2")

    # 在最大值点添加标记
    ax.scatter(max_z_x, max_z_y, max_z, color='red', label='Max Beta Point')

    # 添加文字注释
    ax.text(max_z_x, max_z_y, max_z, f'Max Beta: {max_z:.2f}\n'
                                     f'Delta Mass: {max_z_x:.2f}\n'
                                     f'Velocity: {max_z_y:.2f}', color='red')

    plt.show()

# plot_result(m, v)

plot_3D(m, v)

# 仅用于验证
# _plot_3(m, v)

# 仅用于_plot_2
# m = np.arange(0, 100000, 10000)
# v = np.array(1)