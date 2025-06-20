import numpy as np
import torch
def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    train_labels = torch.from_numpy(train_labels)
    n = train_labels.shape[0]
    K = train_labels.shape[1]

    partialY = train_labels

    transition_matrix = train_labels.numpy().copy()
    transition_matrix = np.where(transition_matrix != 1, partial_rate, transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[j, :]) * 1)

    partialY = partialY.numpy()

    print("Finish Generating Candidate Label Sets!\n")
    return partialY