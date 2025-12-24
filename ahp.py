import numpy as np

def ahp(pairwise_matrix):
    eigvals, eigvecs = np.linalg.eig(pairwise_matrix)
    max_index = np.argmax(eigvals.real)

    weights = eigvecs[:, max_index].real
    weights = weights / weights.sum()

    n = pairwise_matrix.shape[0]
    CI = (eigvals[max_index].real - n) / (n - 1)

    RI = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12}
    CR = CI / RI[n]

    return weights, CI, CR
