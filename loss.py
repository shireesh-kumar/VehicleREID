import torch
import torch.nn.functional as F

def trihard_loss(embeddings, labels, margin=1.0):
    batch_size = embeddings.shape[0]
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # Compute pairwise distances

    loss = 0.0
    for i in range(batch_size):
        anchor_label = labels[i]

        # Positive selection: the farthest positive (same class)
        positive_distances = dist_matrix[i][labels == anchor_label]
        hardest_positive = positive_distances.max()

        # Negative selection: the closest negative (different class)
        negative_distances = dist_matrix[i][labels != anchor_label]
        hardest_negative = negative_distances.min()

        # Apply triplet loss with margin
        loss += F.relu(hardest_positive - hardest_negative + margin)

    return loss / batch_size
