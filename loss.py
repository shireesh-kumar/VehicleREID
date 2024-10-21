import torch
import torch.nn.functional as F

def trihard_loss(embeddings, labels, dc_module, margin=1.0):
    batch_size = embeddings.shape[0]
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # Compute pairwise distances

    loss = torch.tensor(0.0, device=embeddings.device)  # Initialize loss as a tensor on the same device

    for i in range(batch_size):
        anchor_label = labels[i]

        # Positive selection: the hardest positive (same class)
        positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
        if len(positive_indices) > 1:  # More than one positive
            hardest_positive_idx = positive_indices[positive_distances.argmax()]
            hardest_positive = embeddings[hardest_positive_idx]
        else:
            hardest_positive = embeddings[i]  # Default to the anchor if no other positives

        # Negative selection: the hardest negative (different class)
        negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
        if len(negative_indices) > 0:  # There is at least one negative
            hardest_negative_idx = negative_indices[negative_distances.argmin()]
            hardest_negative = embeddings[hardest_negative_idx]
        else:
            hardest_negative = embeddings[i]  # Default to the anchor if no negatives

        # Apply the DC module to refine the embeddings
        refined_positive, refined_negative = dc_module(embeddings[i].unsqueeze(0), hardest_positive.unsqueeze(0), hardest_negative.unsqueeze(0))

        # Compute distances after refinement
        pos_dist = F.pairwise_distance(embeddings[i].unsqueeze(0), refined_positive)
        neg_dist = F.pairwise_distance(embeddings[i].unsqueeze(0), refined_negative)

        # Apply triplet loss with margin
        loss += F.relu(pos_dist - neg_dist + margin)

    return loss / batch_size  # Return the average loss
