import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def Problem_2():
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


def Theoretical_Solve(K, M, plot_flag=False):

    eigen_matrix = M.inv() * K
    eigen_matrix = np.array(eigen_matrix, dtype=float)

    eigen_vals, eigen_vetors = np.linalg.eig(eigen_matrix)

    eigen_vals = np.sqrt(eigen_vals)

    if plot_flag:
        print("Modal Matrix After Orthogonalization:")
        sp.pprint(_to_1(eigen_vetors))
        print("\nMain Frequency:")
        sp.pprint(list(eigen_vals))
        plot_mode_shape_diagram(eigen_vetors)

    return eigen_vetors


def plot_mode_shape_diagram(phi):
    phi = np.array(phi)
    fig, axs = plt.subplots(phi.shape[0])
    for i in range(phi.shape[0]):
        x = range(phi.shape[0] + 2)
        y = phi[:, i].tolist()
        y = [0] + y + [0]
        axs[i].plot(x, y)
        axs[i].plot([0, len(x) - 1], [0, 0])
        axs[i].set_title(f'Mode Shape Diagram {i + 1}:')

    plt.tight_layout()
    plt.show()


def Matrix_Iteration(K, M, phi=[], error=1e-5, epoch_num=15):
    if phi == []:
        phi = sp.ones(M.shape[0], M.shape[1])
        # phi = sp.randMatrix(r=M.shape[0], c=M.shape[1])
        # sp.pprint(phi)

    iiter = sp.zeros(1, M.shape[0])
    omega = sp.zeros(1, M.shape[0])
    lamda = sp.zeros(1, M.shape[0])

    if _is_full_rank(K):
        F = K.inv()

    D = F * M

    for i in range(M.shape[0]):
        phi_i = phi[:, i]
        if i > 0:
            phi_1 = sp.Matrix(phi[:, [i - 1]])
            lamda_1 = sp.Matrix([lamda[i - 1]])
            m_p1 = phi_1.transpose() * M * phi_1
            D = D - (float(lamda_1[0]) / float(m_p1[0])) * phi_1 * phi_1.transpose() * M
        # sp.pprint(D)
        phi_i, omega_i, lamda_i, epoch_i = _iteration(D, phi_i, error, epoch_num)
        phi[:, i] = phi_i[:]
        omega[i] = omega_i
        lamda[i] = lamda_i
        iiter[i] = epoch_i
    # phi = np.array(phi, dtype=float)
    # omega = np.array(omega, dtype=float)
    print("\nModel Matrix:")
    sp.pprint(phi)
    print("\nMain Frequency:")
    sp.pprint(omega)
    print("\nEpoch Numbers:")
    sp.pprint(iiter)


def _iteration(D, phi_0, error, epoch_num):
    epoch = 1
    e = 1e5
    while e >= error and epoch_num >= epoch:
        maximun_0 = max(abs(phi_0))
        phi_1 = D * phi_0
        maximun_1 = max(abs(phi_1))
        phi_1 = phi_1 / maximun_1
        e = np.linalg.norm((phi_1 - phi_0), ord=1)
        phi_0 = phi_1
        epoch += 1
    phi_1 = phi_1 * maximun_1
    lamda = float(phi_1[-1] / (phi_0 * maximun_0)[-1])
    omega = np.sqrt(1 / lamda)
    return phi_0, omega, lamda, epoch


def _is_full_rank(M):
    M = np.array(M, dtype=np.float64)
    if np.linalg.matrix_rank(M) == M.shape[0]:
        return True
    else:
        return False


def Sub_Space_Matrix_Iteration(K: sp.Matrix, M: sp.Matrix, D: sp.Matrix = sp.Matrix([]),
                               phi: sp.Matrix = sp.Matrix([]), last_phi: sp.Matrix = sp.Matrix([]),
                               r=None, error=1e-3, epoch_num=30):
    _phi = phi - last_phi
    if epoch_num == 0:
        print("Out of epoch_num")
        print("Ans:\n")
        sp.pprint(last_phi)
        return last_phi
    elif _phi != sp.Matrix([]) and _phi.norm(1) <= error:
        print("Ans:\n")
        sp.pprint(last_phi)
        return last_phi

    if phi == sp.Matrix([]):
        if r == None:
            r = M.shape[0] - 1
        phi = sp.randMatrix(M.shape[0], r)
        phi = _to_1(phi)
        last_phi = phi.copy()
        if _is_full_rank(K):
            F = K.inv()
            D = F * M

    last_phi = phi.copy()
    phi = D * phi
    phi = _gram_schmidt(phi)
    _K = phi.transpose() * K * phi
    _M = phi.transpose() * M * phi

    _phi = Theoretical_Solve(_K, _M)
    phi = phi * _phi
    phi = _to_1(phi)
    phi = Sub_Space_Matrix_Iteration(K, M, D, phi, last_phi, r, error, epoch_num - 1)


def _to_1(M):
    M = np.array(M, dtype=np.float32)
    M = M / np.max(np.abs(M), axis=0)
    M = sp.Matrix(M)
    return M


def _gram_schmidt(matrix):

    matrix = np.array(matrix, dtype=float)
    # 获取矩阵的行数和列数
    num_rows, num_cols = matrix.shape

    # 创建一个空矩阵，用于存储正交化后的向量
    orthogonal_matrix = np.zeros((num_rows, num_cols))

    # 对第一个向量直接进行单位化
    orthogonal_matrix[:, 0] = matrix[:, 0] / np.linalg.norm(matrix[:, 0])

    # 对剩余的向量进行施密特正交化
    for i in range(1, num_cols):
        # 计算投影
        projection = np.zeros(num_rows)
        for j in range(i):
            projection += np.dot(matrix[:, i], orthogonal_matrix[:, j]) * orthogonal_matrix[:, j]

        # 计算正交向量
        orthogonal_matrix[:, i] = matrix[:, i] - projection

        # 单位化正交向量
        orthogonal_matrix[:, i] /= np.linalg.norm(orthogonal_matrix[:, i])
    orthogonal_matrix = orthogonal_matrix / np.max(orthogonal_matrix, axis=0)
    orthogonal_matrix = sp.Matrix(orthogonal_matrix)
    return orthogonal_matrix


if __name__ == '__main__':

    M = sp.Matrix([[1, 0], [0, 2]])
    K = sp.Matrix([[2, -1], [-1, 3]])

    M = sp.Matrix([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]])
    K = sp.Matrix([[4,-1,-1,-1],
                  [-1,3,-1,0],
                  [-1,-1,4,-1],
                  [-1,0,-1,3]])
    """
    M = sp.Matrix([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 2]])
    K = sp.Matrix([[2, -1, 0],
                   [-1, 3, -2],
                   [0, -2, 2]])
    
    M = sp.Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
    K = sp.Matrix([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    """
    Theoretical_Solve(K, M, True)
    Matrix_Iteration(K, M)
    Sub_Space_Matrix_Iteration(K, M)