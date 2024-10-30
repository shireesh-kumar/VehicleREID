from __future__ import absolute_import
from __future__ import division


import torch
import torch.nn.functional as F
from dc_module import DCModuleOptimized
import torch.profiler
import torch.nn as nn



class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.dc_module = DCModuleOptimized()
            
    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, channels, w , h )
        - targets: ground truth labels with shape (batch_size)
        """
        b,c,w,h = inputs.size() 
        inputs = inputs.view(b, -1)  # Now it will be (batch_size, 4096)
        n = inputs.size(0) # batch_size, batch_size
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # dist_ap, dist_an = [], []
        hard_positive_images, hard_negative_images = [], []


        for i in range(n):
            # Hardest positive
            hard_pos_indices = mask[i].nonzero(as_tuple=True)[0]
            hardest_positive_index = hard_pos_indices[dist[i][hard_pos_indices].argmax()]
            # dist_ap.append(dist[i][hardest_positive_index].unsqueeze(0))
            hard_positive_images.append(inputs[hardest_positive_index].unsqueeze(0))

            # Hardest negative
            hard_neg_indices = (mask[i] == 0).nonzero(as_tuple=True)[0]
            hardest_negative_index = hard_neg_indices[dist[i][hard_neg_indices].argmin()]
            # dist_an.append(dist[i][hardest_negative_index].unsqueeze(0))
            hard_negative_images.append(inputs[hardest_negative_index].unsqueeze(0))

        h_pos_i = torch.stack(hard_positive_images).squeeze(1).view(b,w,h)
        h_neg_i = torch.stack(hard_negative_images).squeeze(1).view(b,w,h)
        inputs = inputs.view(b,w,h)


        # dist_ap = torch.cat(dist_ap)
        # dist_an = torch.cat(dist_an)
        # Initialize lists to store modified positives and negatives
        modi_pos_list = []
        modi_neg_list = []

        # Iterate through the triplets and process them individually
        for anchor, hard_pos, hard_neg in zip(inputs, h_pos_i, h_neg_i):
            modi_pos, modi_neg = self.dc_module(anchor, hard_pos, hard_neg)
            modi_pos_list.append(modi_pos)
            modi_neg_list.append(modi_neg)

        modi_pos_tensor = torch.stack(modi_pos_list)
        modi_neg_tensor = torch.stack(modi_neg_list)
        

        # Flatten the inputs, hard positives, and hard negatives for distance computation
        inputs_flat = inputs.view(inputs.size(0), -1)  # Shape: (batch_size, channels * w * h)
        h_pos_flat = modi_pos_tensor.view(h_pos_i.size(0), -1)  # Shape: (batch_size, channels * w * h)
        h_neg_flat = modi_neg_tensor.view(h_neg_i.size(0), -1)  # Shape: (batch_size, channels * w * h)
        
        # Compute distances to hard positives (dist_ap) and hard negatives (dist_an)
        dist_ap = (inputs_flat - h_pos_flat).pow(2).sum(dim=1).sqrt()  # Element-wise operation
        dist_an = (inputs_flat - h_neg_flat).pow(2).sum(dim=1).sqrt()  # Element-wise operation

        # Since we already have the hard positives and negatives, we can compute the loss directly
        y = torch.ones_like(dist_an)  # Target labels for loss
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss  # Return the computed loss

    

# # Set random seed for reproducibility
# torch.manual_seed(42)

# # Define batch size and image dimensions
# batch_size = 32
# channels = 1
# width = 64
# height = 64
# num_classes = 5

# # Generate random inputs and labels
# inputs = torch.randn(batch_size, channels, width, height)  # Shape: (32, 1, 64, 64)
# labels = torch.randint(0, num_classes, (batch_size,))      # Random labels for each sample

# # Initialize the TripletLoss
# margin = 1.0
# triplet_loss_fn = TripletLoss(margin=margin)

# # Calculate the loss
# loss = triplet_loss_fn(inputs, labels)

# print("Triplet Loss:", loss.item())


