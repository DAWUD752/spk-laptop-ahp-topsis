import numpy as np

def topsis(data, weights, impacts):
    X = data.values.astype(float)

    # Normalisasi
    norm = X / np.sqrt((X ** 2).sum(axis=0))

    # Matriks ternormalisasi terbobot
    weighted = norm * weights

    # Solusi ideal positif & negatif
    ideal_pos = np.where(impacts == 1, weighted.max(axis=0), weighted.min(axis=0))
    ideal_neg = np.where(impacts == 1, weighted.min(axis=0), weighted.max(axis=0))

    # Jarak ke solusi ideal
    d_pos = np.sqrt(((weighted - ideal_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - ideal_neg) ** 2).sum(axis=1))

    # Nilai preferensi
    return d_neg / (d_pos + d_neg)
