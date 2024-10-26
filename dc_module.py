import torch
import torch.nn as nn

class DCModule(nn.Module):
    def __init__(self, window_size=3, step_size=2):
        super(DCModule, self).__init__()
        self.window_size = window_size
        self.step_size = step_size

    def forward(self, anchor, positive, negative):  

        return (
            self.pool(anchor, positive, min) + self.pool(anchor, positive, max),
            self.pool(anchor, negative, min) + self.pool(anchor, negative, max)
        )
    
    def pool(self, anchor, comparison, reduction_fn):
        h, w = anchor.shape
        refined_map = comparison.clone()

        for i in range(0, h - self.window_size + 1, self.step_size):
            for j in range(0, w - self.window_size + 1, self.step_size):
                anchor_patch = anchor[i:i + self.window_size, j:j + self.window_size]
                comp_patch = comparison[i:i + self.window_size, j:j + self.window_size]
                diff = torch.abs(anchor_patch - comp_patch)
                selected_pixel_idx = diff.reshape(-1).argmax() if reduction_fn == max else diff.reshape(-1).argmin()
                refined_map[i:i + self.window_size, j:j + self.window_size] = comp_patch.reshape(-1)[selected_pixel_idx]

        return refined_map
