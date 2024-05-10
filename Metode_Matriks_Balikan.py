import numpy as np

def solve_linear_system(A, b):
    """
    Mencari solusi sistem persamaan linear Ax = b menggunakan metode matriks balikan.

    Parameter:
    A : numpy array
        Matriks koefisien
    b : numpy array
        Vektor hasil

    Return:
    x : numpy array
        Solusi sistem persamaan linear
    """
    if np.linalg.matrix_rank(A) != A.shape[0]:
        print("Sistem persamaan tidak memiliki solusi atau solusi tak tunggal.")
        return None
    
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    return x

# Kode Testing
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

x = solve_linear_system(A, b)
if x is not None:
    print("Solusi sistem persamaan linear:")
    print(x)
