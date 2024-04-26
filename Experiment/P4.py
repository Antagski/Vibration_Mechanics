import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch
from torch import pow, sqrt


def _f(s: torch.tensor, alpha: torch.tensor = torch.tensor([1.0], dtype=torch.float64),
       zeta: torch.tensor = torch.tensor([0.5], dtype=torch.float64), miu=0.2) -> torch.tensor:
    if s.shape != torch.Size([]):
        # alpha = torch.tensor(alpha, dtype=torch.float64)
        alpha = alpha[0].repeat(s.shape[0])
        # zeta = torch.tensor(zeta, dtype=torch.float64)
        zeta = zeta[0].repeat(s.shape[0])
    y = (sqrt(pow((pow(s, 2) - pow(alpha, 2)), 2) + pow((2 * zeta * s), 2)) /
         (sqrt(pow((miu * pow(s, 2) * pow(alpha, 2) - (pow(s, 2) - 1) * (pow(s, 2) - pow(alpha, 2))), 2) +
               pow((2 * zeta * s), 2) * pow((pow(s, 2) - 1 + miu * pow(s, 2)), 2))))
    return y


def _f_partial(s: torch.tensor, alpha=torch.tensor([1.0]), zeta=torch.tensor([0.5]), miu=0.2) -> torch.tensor:
    if s.shape != torch.Size([]):
        # alpha = torch.tensor(alpha, dtype=torch.float64)
        alpha = alpha[0].repeat(s.shape[0])
        # zeta = torch.tensor(zeta, dtype=torch.float64)
        zeta = zeta[0].repeat(s.shape[0])
    numerator = (-2 * s * (s - alpha) * (s + alpha) * ((s ** 2 - alpha ** 2) ** 2 + alpha ** 4 * miu) *
                 (s ** 4 + alpha ** 2 - s ** 2 * (1 + alpha ** 2 * (1 + miu))) -
                 4 * s ** 3 * (-4 * alpha ** 4 * (1 + miu) +
                               2 * s ** 2 * alpha ** 2 * (4 + 3 * miu + 2 * alpha ** 2 * (1 + miu) ** 2) +
                               s ** 6 * (4 + miu * (2 + miu)) -
                               4 * s ** 4 * (1 + alpha ** 2 * (1 + miu) * (2 + miu))) * zeta ** 2 -
                 32 * s ** 5 * (1 + miu) * (-1 + s ** 2 * (1 + miu)) * zeta ** 4)
    denominator = (torch.sqrt((s ** 2 - alpha ** 2) ** 2 + 4 * s ** 2 * zeta ** 2) *
                   ((s ** 4 + alpha ** 2 - s ** 2 * (1 + alpha ** 2 * (1 + miu))) ** 2 +
                    4 * s ** 2 * (-1 + s ** 2 * (1 + miu)) ** 2 * zeta ** 2) ** (3 / 2))
    result = numerator / denominator

    # print(result)
    result = torch.norm(result, p=float('inf'))
    # result = torch.sum(result.pow(2))
    return result


def _f_sp(s=sp.Matrix([]), alpha: float = 1.0, zeta: float = 0.0, miu: float = 0.2):
    if s == sp.Matrix([]):
        s = sp.symbols('s')
    y = (sp.sqrt((s ** 2 - alpha ** 2) ** 2 + (2 * zeta * s) ** 2) /
         sp.sqrt((miu * s ** 2 * alpha ** 2 - (s ** 2 - 1) * (s ** 2 - alpha ** 2)) ** 2
                 + (2 * zeta * s) ** 2 * (s ** 2 - 1 + miu * s ** 2) ** 2))
    return y


def _find_point(alpha=1.0, zeta=0.5, miu=0.5):
    y_1 = _f_sp(alpha=alpha, zeta=0, miu=miu)
    y_2 = _f_sp(alpha=alpha, zeta=zeta, miu=miu)
    expression = y_1 - y_2
    solutions = sp.solve(expression, 's')
    solutions = [solution for solution in list(solutions) if
                 isinstance(solution, sp.core.numbers.Float) and solution > 0.01]

    # print(solutions)
    return solutions


def loss(x, alpha: torch.tensor = torch.tensor([1.0]), zeta: torch.tensor = torch.tensor([0.5]),
         miu=0.05, objection='alpha'):
    if objection == 'alpha':
        y = _f(x, alpha=alpha, zeta=zeta, miu=miu)
        l = torch.norm(y - torch.mean(y), p=2)
        # print(f"y {y}")
        # print(f"l {l}")
    elif objection == 'zeta':
        l = _f_partial(x, alpha=alpha, zeta=zeta, miu=miu)
    else:
        print("Wrong Format")
        return None
    return l


def find_zeta(alpha, miu):
    zeta = [torch.tensor(0.1, requires_grad=True)]
    alpha = torch.tensor([alpha])
    lr_zeta = 1e-5
    epoch_num = 300
    l_data = []
    _alpha = alpha[0].detach().numpy()
    _zeta = zeta[0].detach().numpy()
    x_data = torch.tensor(_find_point(alpha=_alpha, zeta=_zeta, miu=miu), dtype=torch.float64)

    for epoch in range(epoch_num):
        print(f"Optimization rounds: {epoch + 1}")

        trainer = torch.optim.SGD(zeta, lr=lr_zeta)
        # trainer = torch.optim.Adam(zeta, lr=1e-5)
        # trainer = torch.optim.Adadelta(zeta, lr=1e-4)
        l = loss(x_data, alpha, zeta, miu=miu, objection='zeta')

        trainer.zero_grad()
        l.backward()
        trainer.step()

        # l = loss(x_data, alpha, zeta, objection='zeta')

        l_data.append(l.detach().numpy())

    plotting(alpha=alpha, zeta=zeta, miu=miu, loss=l_data)


def find_alpha(zeta, miu):
    alpha = [torch.tensor(0.5, requires_grad=True)]
    zeta = torch.tensor([zeta])
    lr_alpha = 0.005
    epoch_num = 30
    l_data = []

    for epoch in range(epoch_num):

        print(f"Optimization rounds: {epoch + 1}")

        if epoch % 10 == 0:
            lr_alpha *= 0.8

        trainer = torch.optim.SGD(alpha, lr=lr_alpha)
        # trainer = torch.optim.Adam(alpha, lr=1e-3)
        _alpha = alpha[0].detach().numpy()
        _zeta = zeta[0].detach().numpy()
        x_data = torch.tensor(_find_point(alpha=_alpha, zeta=_zeta, miu=miu), dtype=torch.float64)
        l = loss(x_data, alpha=alpha, zeta=zeta, miu=miu)

        trainer.zero_grad()
        l.backward()
        trainer.step()

        l_data.append(l.detach().numpy())

    plotting(alpha=alpha, zeta=zeta, miu=miu, loss=l_data)


def plotting(alpha, zeta, miu, loss, mode='alpha'):
    fig, axs = plt.subplots(2, 1)

    s = np.linspace(0.5, 1.5, 1000)
    s = torch.tensor(s, dtype=torch.float64)

    axs[1].plot(loss)
    axs[1].set_title("Iteration Curve")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")

    y_1 = _f(s, alpha=alpha, zeta=torch.tensor([0.0]), miu=miu)
    y_1 = y_1.detach().numpy()
    y_2 = _f(s, alpha=alpha, zeta=zeta, miu=miu)
    y_2 = y_2.detach().numpy()
    s = np.linspace(0.5, 1.5, 1000)
    zeta = zeta[0].detach().numpy()
    alpha = alpha[0].detach().numpy()
    axs[0].plot(s, y_1, label='zeta=0.00')
    axs[0].plot(s, y_2, label=f'zeta={zeta:.2f}')
    axs[0].legend()
    axs[0].set_ylim(0, max(y_2) + 2)

    if mode == 'alpha':
        plt.suptitle('Alpha Optimization')
    else:
        plt.suptitle('Zeta Optimization')

    print(f"Alpha: {alpha}\nZeta: {zeta}")

    plt.show()


if __name__ == "__main__":
    find_alpha(zeta=0.5, miu=0.2)
    find_zeta(alpha=1.0, miu=0.2)
