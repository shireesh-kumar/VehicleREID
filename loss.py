import torch
import torch.nn.functional as F

def trihard_loss(embeddings, labels, dc_module, margin=1.0):
    print("Reached Loss Computation Method")
    
    # Check the shape of embeddings and ensure they are (batch_size, embedding_dim)
    if len(embeddings.shape) == 4:
        embeddings = embeddings.squeeze(1)  # Remove the channel dimension, shape becomes (batch_size, 128, 128)

    batch_size = embeddings.shape[0]
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(embeddings.view(batch_size, -1), embeddings.view(batch_size, -1), p=2)  # (batch_size, 16384)
        
    loss = torch.tensor(0.0, device=embeddings.device)  # Initialize loss as a tensor on the same device


    for i in range(batch_size):
        anchor_label = labels[i]

        # Positive selection: the hardest positive (same class)
        positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
        positive_indices = positive_indices.to(dist_matrix.device)

        if len(positive_indices) > 1:  # More than one positive
            # Compute positive distances
            positive_distances = dist_matrix[i, positive_indices]
            hardest_positive_idx = positive_indices[positive_distances.argmin()]  # Choose the hardest positive
            hardest_positive = embeddings[hardest_positive_idx]
        else:
            hardest_positive = embeddings[i]  # Default to the anchor if no other positives

        # Negative selection: the hardest negative (different class)
        negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
        negative_indices = negative_indices.to(dist_matrix.device)

        if len(negative_indices) > 0:  # There is at least one negative
            # Compute negative distances
            negative_distances = dist_matrix[i, negative_indices]
            hardest_negative_idx = negative_indices[negative_distances.argmax()]  # Choose the hardest negative
            hardest_negative = embeddings[hardest_negative_idx]
        else:
            hardest_negative = embeddings[i]  # Default to the anchor if no negatives


        # Apply the DC module to refine the embeddings
        refined_positive, refined_negative = dc_module(
            embeddings[i], 
            hardest_positive, 
            hardest_negative
        )
        N = embeddings[i].numel()  # Total number of elements (pixels) in the embedding

        # Compute distances after refinement
        pos_dist = torch.norm(embeddings[i] - refined_positive, p=2) / N  # Total distance to the hardest positive
        neg_dist = torch.norm(embeddings[i] - refined_negative, p=2) / N # Total distance to the hardest negative

        # Apply triplet loss with margin
        loss += F.relu(pos_dist - neg_dist + margin) #margin smaller 

    del dist_matrix, positive_indices, negative_indices, hardest_positive, hardest_negative, refined_positive, refined_negative, pos_dist, neg_dist
    torch.cuda.empty_cache() 


    return loss / batch_size  # Return the average loss
