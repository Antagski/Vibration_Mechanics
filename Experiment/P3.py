import sympy as sp
from Calculator import Theoretical_Solve, Matrix_Iteration, Sub_Space_Matrix_Iteration


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
    Sub_Space_Matrix_Iteration(K, M)
