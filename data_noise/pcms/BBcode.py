import numpy as np



def bbpcm(
    l : int,
    m : int,
    A_poly : list,
    B_poly : list
):
    """
    Creates a parity check matrix for a bivariate bicycle code.

    The function implements the construction of a parity check matrix for a bivariate bicycle code, 
    as described in the paper (https://arxiv.org/pdf/2308.07915.pdf).

    Parameters:
    - l (int): The first dimension parameter of the bicycle code.
    - m (int): The second dimension parameter of the bicycle code.
    - A_poly (list): List of tuples representing powers of x in the A matrix polynomial.
                     Each tuple should be of the form (0 or 1, power).
                     0 corresponds to x, 1 corresponds to y.
    - B_poly (list): List of tuples representing powers of x in the B matrix polynomial.
                     Each tuple should be of the form (0 or 1, power).
                     0 corresponds to x, 1 corresponds to y.

    Returns:
    - H (numpy.ndarray): The resulting parity check matrix.
    - Hx (numpy.ndarray): The left part of the parity check matrix. Which we consider for X-error detection.
    - Hz (numpy.ndarray): The right part of the parity check matrix. Which we consider for Z-error detection.

    Note: We make Hx to detect X-errors and Hz to detect Z-errors omitting the symplectic product. That is, 
    when a quantum error e operates on a CSS QEC code, it does so through the symplectic product. 
    
            (
                0      Hx                           
    H o e =     Hz       0      o (ex | ez)^T     =  (Hx * ex | Hz * ez ) ^T
                            )
    
    In order to ease computations, we will consider the right hand side of the equation directly when
    considering noise decoding. And so we will construct a parity check matrix not in the symplectic form
    to which we can apply the error product directly.
    
            (
                Hx      0                           
    H' =        0       Hz      (ex | ez)^T     =  (Hx * ex | Hz * ez ) ^T
                            )
    """
    
    # We create the two cyclic matrices.
    S_l = np.roll(np.eye(l), 1, axis = 1)
    S_m = np.roll(np.eye(m), 1, axis = 1)
    
    # We compute the x and y variables.
    x = np.kron(S_l, np.eye(m))
    y = np.kron(np.eye(l), S_m)
    
    # Following the input polynomials, we construct A and B.
    A = np.zeros((l*m, l*m))
    for element in A_poly:
        if element[0] == 0:
            A = (A + np.linalg.matrix_power(x, element[1])) % 2
        elif element[0] == 1:
            A = (A + np.linalg.matrix_power(y, element[1])) % 2
    B = np.zeros((l*m, l*m))
    for element in B_poly:
        if element[0] == 0:
            B = (B + np.linalg.matrix_power(x, element[1])) % 2
        elif element[0] == 1:
            B = (B + np.linalg.matrix_power(y, element[1])) % 2

    # Finally, we construct the parity check matrix, Hx and Hz.
    Hx = np.hstack((A, B))
    Hz = np.hstack((B.T, A.T))
    zero_contribution = np.zeros(Hx.shape)
    top = np.hstack((Hx, zero_contribution))
    bottom = np.hstack((zero_contribution, Hz))
    H = np.vstack((top, bottom))
    return H, Hx, Hz

# from scipy.linalg import null_space

# Hx_nullspace = null_space(Hx)
# Hz_nullspace = null_space(Hz)