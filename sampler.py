from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image 


class VehicleDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        Args:
            samples (dict): Dictionary where keys are indices and values are pids.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.samples = samples
        self.transform = transform 
        self.data = {index: pid for pid, indices in samples.items() for index in indices}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Assuming self.data is a list of tuples (pid, image_path)
        pid, image_path = self.data[index],index

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply the transform if provided
        if self.transform:
            image = self.transform(image)

        return image, pid


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source.data
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for (index, pid) in self.data_source.items():
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(self.pids) >= self.num_pids_per_batch:
            # final_idxs = []  # Reset for a new batch
            selected_pids = random.sample(self.pids, self.num_pids_per_batch)
            
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    self.pids.remove(pid)  # Remove PID if no more samples left
            yield from final_idxs  # Use yield to return this batch and continue
        
        # return iter(final_idxs)

    def __len__(self):
        return self.length
