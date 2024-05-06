import sympy as sp
from Calculator import Theoretical_Solve, Matrix_Iteration, Sub_Space_Matrix_Iteration, _to_1


if __name__ == '__main__':
    M = sp.Matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    K = sp.Matrix([[4, -1, -1, -1],
                   [-1, 3, -1, 0],
                   [-1, -1, 4, -1],
                   [-1, 0, -1, 3]])

    Theoretical_Solve(K, M, True)
    Matrix_Iteration(K, M)
    phi = sp.randMatrix(M.shape[0], 4)
    phi = _to_1(phi)
    for r in [2 ,3, 4]:
        _phi = phi[:, 0:r]
        Sub_Space_Matrix_Iteration(K, M, phi=_phi, r=r)
    print("\nAssumed Mode Matrix:")
    sp.pprint(phi)
