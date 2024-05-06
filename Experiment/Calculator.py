import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def Theoretical_Solve(K, M, plot_flag=False):
    """
    理论解求解函数，使用NUMPY库内置特征值函数求解特征向量，
    并使用施密特正交化使得各阶模态正交
    :param K: 刚度矩阵
    :param M: 质量矩阵
    :param plot_flag: 绘图开关，默认False关闭
    :return: 正交化后的模态矩阵及固有频率
    """

    # 求解各阶模态及固有频率
    eigen_matrix = M.inv() * K
    eigen_matrix = np.array(eigen_matrix, dtype=float)

    eigen_vals, eigen_vectors = np.linalg.eig(eigen_matrix)

    eigen_vals = np.sqrt(eigen_vals)
    eigen_vals = np.sort(eigen_vals)


    # 结果可视化
    if plot_flag:
        print("\n#####################\n"
              "# Theoretical Solve #\n"
              "#####################\n")
        print("Modal Matrix After Orthogonalization:")
        # 对模态进行施密特正交化
        sp.pprint(_gram_schmidt(eigen_vectors))
        print("\nMain Frequency:")
        sp.pprint(list(eigen_vals))
        
        plot_mode_shape_diagram(eigen_vectors)

    return eigen_vectors, eigen_vals


def plot_mode_shape_diagram(phi):
    """
    绘制各阶模态的振型图
    :param phi: 模态矩阵
    :return:
    """
    phi = np.array(phi)
    fig, axs = plt.subplots(phi.shape[1])
    for i in range(phi.shape[1]):
        x = range(phi.shape[0] + 2)
        y = phi[:, i].tolist()
        y = [0] + y + [0]
        axs[i].plot(x, y)
        axs[i].plot([0, len(x) - 1], [0, 0])
        axs[i].set_title(f'Mode Shape Diagram {i + 1}:')

    plt.tight_layout()
    plt.show()


def Matrix_Iteration(K, M, phi=[], error=1e-5, epoch_num=15):
    """
    矩阵迭代法求解函数，通过假设模态矩阵进行迭代，
    逐步收敛至真实模态
    :param K: 刚度矩阵
    :param M: 质量矩阵
    :param phi: 假设模态，默认为全一矩阵
    :param error: 精度控制，默认为1e-5
    :param epoch_num: 迭代次数控制，默认为15次
    :return:
    """

    # 初始化假设模态，此处为全一矩阵
    if phi == []:
        phi = sp.ones(M.shape[0], M.shape[1])
        # phi = sp.randMatrix(r=M.shape[0], c=M.shape[1])

    # 初始化各参数容器
    iiter = sp.zeros(1, M.shape[0])
    omega = sp.zeros(1, M.shape[0])
    lamda = sp.zeros(1, M.shape[0])

    # 判断是否可逆
    if _is_full_rank(K):
        F = K.inv()
    else:
        print("Matrix K is NOT invertible")
        return

    # 计算动力矩阵
    D = F * M

    # 矩阵叠加法
    for i in range(M.shape[0]):
        phi_i = phi[:, i]
        if i > 0:
            phi_1 = sp.Matrix(phi[:, [i - 1]])
            lamda_1 = sp.Matrix([lamda[i - 1]])
            m_p1 = phi_1.transpose() * M * phi_1
            D = D - (float(lamda_1[0]) / float(m_p1[0])) * phi_1 * phi_1.transpose() * M
        # sp.pprint(D)
        phi_i, omega_i, lamda_i, epoch_i = _iteration(D, phi_i, error, epoch_num)

        # 更新参数容器
        phi[:, i] = phi_i[:]
        omega[i] = omega_i
        lamda[i] = lamda_i
        iiter[i] = epoch_i

    # 结果可视化
    print("\n##########################\n"
          "# Matrix Iteration Solve #\n"
          "##########################\n")
    print("\nModel Matrix:")
    sp.pprint(phi)
    print("\nMain Frequency:")
    omega = list(omega)
    omega.sort()
    sp.pprint(omega)
    # plot_mode_shape_diagram(phi)
    print("\nEpoch Numbers:")
    sp.pprint(iiter)


def _iteration(D, phi_0, error, epoch_num):
    """
    矩阵迭代法中迭代函数
    :param D: 动力矩阵
    :param phi_0: 第i阶假设模态
    :param error: 误差控制
    :param epoch_num: 迭代次数控制
    :return: 迭代后的假设模态phi_i,第i阶固定频率omega_i,第i阶特征值lamda_i,总迭代次数epoch_i
    """

    # 初始化终止参数
    epoch = 1
    e = 1e5

    # 矩阵迭代
    while e >= error and epoch_num >= epoch:
        maximum_0 = max(abs(phi_0))
        phi_1 = D * phi_0
        maximum_1 = max(abs(phi_1))
        phi_1 = phi_1 / maximum_1
        e = np.linalg.norm((phi_1 - phi_0), ord=1)
        phi_0 = phi_1
        epoch += 1
    phi_1 = phi_1 * maximum_1
    lamda = float(phi_1[-1] / (phi_0 * maximum_0)[-1])
    omega = np.sqrt(1 / lamda)
    return phi_0, omega, lamda, epoch


def _is_full_rank(M):
    """
    判断矩阵是否正交
    :param M: 待判断矩阵
    :return: 布尔值
    """
    M = np.array(M, dtype=np.float64)
    if np.linalg.matrix_rank(M) == M.shape[0]:
        return True
    else:
        return False


def Sub_Space_Matrix_Iteration(K: sp.Matrix, M: sp.Matrix, D: sp.Matrix = sp.Matrix([]),
                               phi: sp.Matrix = sp.Matrix([]), last_phi: sp.Matrix = sp.Matrix([]),
                               omega: sp.Matrix = sp.Matrix([]), r=None, error=1e-3, epoch_num=30):
    """
    子空间迭代法，给出假设模态，通过递推的方式迭代，
    从而逼近真实模态
    :param K: 刚度矩阵
    :param M: 质量矩阵
    :param D: 动力矩阵，默认为空，仅用于迭代过程传递参数
    :param phi: 假设模态，默认为空
    :param last_phi: 上一次迭代中的模态
    :param omega: 固有频率
    :param r: 假设模态的阶数，默认为n-1阶
    :param error: 误差控制，默认为1e-3
    :param epoch_num: 迭代次数控制，默认为30次
    :return:
    """

    if last_phi == sp.Matrix([]):
        _phi = sp.Matrix([])
    else:
        _phi = phi - last_phi

    # 递推终止条件判断
    if epoch_num == 0:

        print("\n####################################\n"
              "# Sub Space Matrix Iteration Solve #"
              "\n####################################\n")
        print("Model Matrix: ")
        sp.pprint(last_phi)
        print("\nOut of epoch_num")
        print("\nMain Frequency:")
        sp.pprint(list(omega))

        # plot_mode_shape_diagram(last_phi)
        
        return last_phi
    elif _phi != sp.Matrix([]) and _phi.norm(1) <= error:
        print("\n####################################\n"
              "# Sub Space Matrix Iteration Solve #"
              "\n####################################\n")
        print("Model Matrix: ")
        sp.pprint(last_phi)
        print(f"\nEpoch Numbers: {30 - epoch_num}")
        print("\nMain Frequency:")
        sp.pprint(list(omega))
        # plot_mode_shape_diagram(last_phi)
        return last_phi

    # 初始化假设模态
    if phi == sp.Matrix([]):
        if r == None:
            r = M.shape[0] - 1
        phi = sp.randMatrix(M.shape[0], r)
        # phi = sp.ones(M.shape[0], r)
        phi = _to_1(phi)
        last_phi = phi.copy()

    if D == sp.Matrix([]):
        if _is_full_rank(K):
            F = K.inv()
            D = F * M
        else:
            print("Matrix K is NOT invertible")
            return

    # 更新迭代参数
    last_phi = phi.copy()
    phi = D * phi
    phi = _gram_schmidt(phi)
    _K = phi.transpose() * K * phi
    _M = phi.transpose() * M * phi

    # 求解缩减后的理论解
    _phi, omega = Theoretical_Solve(_K, _M)
    phi = phi * _phi
    phi = _to_1(phi)

    # 子空间迭代
    phi = Sub_Space_Matrix_Iteration(K, M, D, phi, last_phi, omega, r, error, epoch_num - 1)


def _to_1(M):
    """
    归一化函数，将传入矩阵进行归一化
    :param M: 待归一化矩阵
    :return: 归一化后的矩阵
    """
    M = np.array(M, dtype=np.float32)
    M = M / np.max(np.abs(M), axis=0)
    M = sp.Matrix(M)
    return M


def _gram_schmidt(matrix):
    """
    施密特正交化，将矩阵进行正交化
    :param matrix: 待正交化矩阵
    :return: 正交化后矩阵
    """
    matrix = np.array(matrix, dtype=float)
    num_rows, num_cols = matrix.shape
    orthogonal_matrix = np.zeros((num_rows, num_cols))
    orthogonal_matrix[:, 0] = matrix[:, 0] / np.linalg.norm(matrix[:, 0])

    # 对剩余的向量进行施密特正交化
    for i in range(1, num_cols):
        projection = np.zeros(num_rows)
        for j in range(i):
            projection += np.dot(matrix[:, i], orthogonal_matrix[:, j]) * orthogonal_matrix[:, j]

        orthogonal_matrix[:, i] = matrix[:, i] - projection

        # 单位化正交向量
        orthogonal_matrix[:, i] /= np.linalg.norm(orthogonal_matrix[:, i])
    orthogonal_matrix = orthogonal_matrix / np.max(orthogonal_matrix, axis=0)
    orthogonal_matrix = sp.Matrix(orthogonal_matrix)
    return orthogonal_matrix


if __name__ == '__main__':
    """
    M = sp.Matrix([[1, 0], [0, 2]])
    K = sp.Matrix([[2, -1], [-1, 3]])
    """
    M = sp.Matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    K = sp.Matrix([[4, -1, -1, -1],
                   [-1, 3, -1, 0],
                   [-1, -1, 4, -1],
                   [-1, 0, -1, 3]])
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
