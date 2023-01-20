import numpy as np

transition_matrix = np.zeros((10, 10))
for i in range(10):
    transition_matrix[i, i+1:] = 1/(10-i)

val_vec = np.sum(np.linalg.inv(np.eye(10) - transition_matrix), axis=1)

print(f"Expected number of hops to reach other side: {val_vec[0]}")