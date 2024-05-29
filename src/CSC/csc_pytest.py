import scipy
import scipy.sparse

matrix = [
            [0, 0, 0, 0, 1],
            [5, 8, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 6, 0, 0, 1],
            [0, 0, 0, 7, 0]
        ]

csc_mat = scipy.sparse.csc_matrix(matrix)

print(csc_mat)
print(csc_mat.count_nonzero())
print(csc_mat.indices)
print(csc_mat.indptr)