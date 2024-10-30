# import torch
# import torch.nn as nn

# class DCModule(nn.Module):
#     def __init__(self, window_size=3, step_size=2):
#         super(DCModule, self).__init__()
#         self.window_size = window_size
#         self.step_size = step_size

#     def forward(self, anchor, positive, negative):  

#         return (
#             self.pool(anchor, positive, min) + self.pool(anchor, positive, max),
#             self.pool(anchor, negative, min) + self.pool(anchor, negative, max)
#         )
    
#     def pool(self, anchor, comparison, reduction_fn):
#         h, w = anchor.shape
#         refined_map = comparison.clone()

#         for i in range(0, h - self.window_size + 1, self.step_size):
#             for j in range(0, w - self.window_size + 1, self.step_size):
#                 anchor_patch = anchor[i:i + self.window_size, j:j + self.window_size]
#                 comp_patch = comparison[i:i + self.window_size, j:j + self.window_size]
#                 diff = torch.abs(anchor_patch - comp_patch)
#                 selected_pixel_idx = diff.reshape(-1).argmax() if reduction_fn == max else diff.reshape(-1).argmin()
#                 refined_map[i:i + self.window_size, j:j + self.window_size] = comp_patch.reshape(-1)[selected_pixel_idx]

#         return refined_map

    

import torch
import torch.nn as nn
import torch.nn.functional as F

class DCModuleOptimized(nn.Module):
    def __init__(self, window_size=3, step_size=2):
        super(DCModuleOptimized, self).__init__()
        self.window_size = window_size
        self.step_size = step_size

    def forward(self, anchor, positive,negative):
        if torch.cuda.is_available():
            # Set up two CUDA streams for parallel computation on GPU
            stream1 = torch.cuda.Stream()
            stream2 = torch.cuda.Stream()

            with torch.cuda.stream(stream1):
                min_pos, max_pos = self.pool(anchor, positive)
            with torch.cuda.stream(stream2):
                min_neg, max_neg = self.pool(anchor, negative)

            # Ensure both streams are complete before proceeding
            torch.cuda.synchronize()

        else:
            # Fallback to sequential computation if CUDA is not available
            min_pos, max_pos = self.pool(anchor, positive)
            min_neg, max_neg = self.pool(anchor, negative)

        return (min_pos + max_pos), (min_neg + max_neg)


    def pool(self, anchor, comparison):
        
        # Using unfold for patch-based extraction
        anchor_patches = F.unfold(anchor.unsqueeze(0), kernel_size=self.window_size, stride=self.step_size).contiguous()
        comparison_patches = F.unfold(comparison.unsqueeze(0), kernel_size=self.window_size, stride=self.step_size).contiguous()

        # Calculate absolute differences in patches
        diff = (anchor_patches - comparison_patches).abs()
        _, min_indices = diff.view(-1, self.window_size * self.window_size).min(dim=1)
        _, max_indices = diff.view(-1, self.window_size * self.window_size).max(dim=1)


        # Reshape comparison_patches for efficient indexing
        min_comparison_patches_2d = comparison_patches.view(-1, self.window_size * self.window_size)
        max_comparison_patches_2d = comparison_patches.view(-1, self.window_size * self.window_size).clone()

        # Directly create a tensor for min values
        min_values_tensor = min_comparison_patches_2d[torch.arange(len(min_indices)), min_indices].view(-1, 1)
        max_values_tensor = max_comparison_patches_2d[torch.arange(len(max_indices)), max_indices].view(-1, 1)

        # Update min/max values in place
        min_comparison_patches_2d[:] = min_values_tensor.expand(-1, min_comparison_patches_2d.size(1))
        max_comparison_patches_2d[:] = max_values_tensor.expand(-1, max_comparison_patches_2d.size(1))


        # Get original height and width
        H, W = comparison.shape

        # Create an empty tensor to hold the reconstructed image
        min_reconstructed_comparison = torch.empty_like(comparison)
        max_reconstructed_comparison = torch.empty_like(comparison)


        # Create indices for the patches
        num_patches_h = (H - self.window_size) // self.step_size + 1
        num_patches_w = (W - self.window_size) // self.step_size + 1

        # Create a grid of all patch indices
        i_indices = torch.arange(num_patches_h) * self.step_size
        j_indices = torch.arange(num_patches_w) * self.step_size
        i_indices, j_indices = torch.meshgrid(i_indices, j_indices, indexing='ij')

        # Calculate the patch indices directly
        for idx in range(num_patches_h * num_patches_w):
            i = i_indices.flatten()[idx]
            j = j_indices.flatten()[idx]
            min_reconstructed_comparison[i:i + self.window_size, j:j + self.window_size] = min_comparison_patches_2d[idx].view(self.window_size, self.window_size)
            max_reconstructed_comparison[i:i + self.window_size, j:j + self.window_size] = max_comparison_patches_2d[idx].view(self.window_size, self.window_size)

        min_reconstructed_comparison = min_reconstructed_comparison.squeeze(0)
        max_reconstructed_comparison = max_reconstructed_comparison.squeeze(0)

        return min_reconstructed_comparison,max_reconstructed_comparison

    
# # Initialize the DCModule
# dc_module = DCModuleOptimized(window_size=2, step_size=1)

# # Create small test matrices (3x3)
# anchor = torch.tensor([[3,5,2], [1,6,4], [7,9,8]], dtype=torch.float32)
# positive = torch.tensor([[2 , 1,2], [ 1, 2,1], [2, 1,2]], dtype=torch.float32)

# negative = torch.tensor([[3,2,1], [9,8,7], [5,4,6]], dtype=torch.float32)

# # Pass through the DCModule
# output_pos , output_neg = dc_module(anchor, positive,negative)                        

# # Print the outputs
# print("Output with Positive Comparison:\n", output_pos)
# print("Output with Negative Comparison:\n", output_neg)
