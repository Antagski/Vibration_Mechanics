import sympy as sp
from Calculator import Theoretical_Solve


if __name__ == '__main__':
    M = sp.Matrix([[1, 0], [0, 2]])
    K = sp.Matrix([[2, -1], [-1, 3]])

    Theoretical_Solve(K, M, True)
