import torch
import torch.nn.functional as F
import torch.nn as nn


class DCModule(nn.Module):
    def __init__(self, window_size=3, step_size=2):
        super(DCModule, self).__init__()
        self.window_size = window_size
        self.step_size = step_size

    def forward(self, anchor, positive, negative):
        refined_positive = self.similarity_pooling(anchor, positive)
        refined_negative = self.difference_pooling(anchor, negative)
        return refined_positive, refined_negative
    
    def similarity_pooling(self, anchor, comparison):
        h, w = anchor.shape
        refined_map = comparison.clone()

        for i in range(0, h - self.window_size + 1, self.step_size):
            for j in range(0, w - self.window_size + 1, self.step_size):
                anchor_patch = anchor[i:i + self.window_size, j:j + self.window_size]
                comp_patch = comparison[i:i + self.window_size, j:j + self.window_size]
                diff = torch.abs(anchor_patch - comp_patch)
                min_diff_idx = torch.argmin(diff.view(-1))
                selected_pixel = comp_patch.view(-1)[min_diff_idx]
                refined_map[i:i + self.window_size, j:j + self.window_size] = selected_pixel

        return refined_map
    
    def difference_pooling(self, anchor, comparison):
        h, w = anchor.shape
        refined_map = comparison.clone()

        for i in range(0, h - self.window_size + 1, self.step_size):
            for j in range(0, w - self.window_size + 1, self.step_size):
                anchor_patch = anchor[i:i + self.window_size, j:j + self.window_size]
                comp_patch = comparison[i:i + self.window_size, j:j + self.window_size]
                diff = torch.abs(anchor_patch - comp_patch)
                max_diff_idx = torch.argmax(diff.view(-1))
                selected_pixel = comp_patch.view(-1)[max_diff_idx]
                refined_map[i:i + self.window_size, j:j + self.window_size] = selected_pixel

        return refined_map