from module import bpbp
import numpy as np

def generate_random_matrix():
    N = np.random.randint(3, 10)  # Random number of rows between 1 and 10
    M = np.random.randint(3, 10)  # Random number of columns between 1 and 10

    # Generate a random NxM matrix with 0s and 1s
    matrix = np.random.randint(2, size=(N, M), dtype=np.uint8)
    return matrix

# Example usage:
mat = generate_random_matrix()

n, m = np.shape(mat)
print("Matrix shape in python is: {}x{}".format(n, m))
print(mat)

#bpbp.koh_v2(mat)


bpotf_obj = bpbp.OBPOTF(mat, 0.01)

bpotf_obj.print_object()

fake_syndrome = np.random.randint(2, size=(n, 1), dtype=np.int32)
print(fake_syndrome)
time = 0.0
error = bpotf_obj.decode(fake_syndrome, time)