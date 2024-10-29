import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms

from data_preprocessing import  get_label_to_samples, set_global_seed, split_label_to_samples
from sampler import VehicleDataset, RandomIdentitySampler

from loss import TripletLoss
from feature_extractor import CustomResNet18
import torch.nn as nn
from torchvision.models import  resnet18, ResNet18_Weights
import torch.distributed as dist  # Importing the distributed module


'''Important'''
set_global_seed(42)



# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"### Using device: {device} ###")
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

trihard_loss = TripletLoss()


label_to_samples = get_label_to_samples()

_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 (width, height)
    transforms.ToTensor(),           # Convert the image to a tensor
    transforms.Normalize(            # Normalize the tensor
    mean=[0.485, 0.456, 0.406],  # Mean for each channel
    std=[0.229, 0.224, 0.225]    # Standard deviation for each channel
    )
])

# Create dataset objects for training and validation
train_label_to_samples, val_label_to_samples = split_label_to_samples(label_to_samples, train_ratio=0.8)

# Create the datasets for training and validation
train_dataset = VehicleDataset(train_label_to_samples, transform=_transform)
val_dataset = VehicleDataset(val_label_to_samples, transform=_transform)

# print("### Below are the train and val dataset lengths ###")
# print(f"Train : {len(train_dataset)}")
# print(f"Val   : {len(val_dataset)}")
            
# Create DataLoader for both train and validation sets
p, k = 2, 16

# print(f"Sampler valeus : P = {p} and K = {k}")
train_sampler = RandomIdentitySampler(train_dataset, train_label_to_samples, batch_size=p*k, num_instances=k)
val_sampler = RandomIdentitySampler(val_dataset, val_label_to_samples, batch_size=p*k, num_instances=k)

train_loader = DataLoader(train_dataset, batch_size=p * k, sampler=train_sampler,num_workers=8,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=p * k, sampler=val_sampler,num_workers=8,pin_memory=True)

# print(f"Train Loader {len(train_loader)}")
# print(f"Train Sampler = {len(train_sampler)}")

# # Create an instance of the DCModule
# dc_module = DCModule(window_size=3, step_size=2).to(device)   # Adjust parameters as needed


# Model definition and optimizer
weights =ResNet18_Weights.IMAGENET1K_V1
model = CustomResNet18(weights=weights).to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop setup
num_epochs = 1
train_losses = []
val_losses = []

# torch.cuda.memory._record_memory_history(
#     max_entries= 1200
# )

print("## Initiating Training ... ##", flush=True)
count = 0
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:

        images = batch[0]  # The first element is the images tensor
        labels = batch[1]  # The second element is the tuple of labels
        
        # Move images and labels to the device
        images = images.to(device)
        labels = labels.to(device) 
        embeddings = model(images)  # Forward pass to get embeddings
        #Compute triplet loss for each unique vehicle ID
        triplet_loss = trihard_loss(embeddings, labels)
        # Backpropagation and optimization
        optimizer.zero_grad(set_to_none=True)
        triplet_loss.backward()          # Backward pass
        optimizer.step()                 # Update the parameters

        total_train_loss += triplet_loss.item()
        print(f"Batch{count+1} completed for Epoch {epoch+1} ", flush=True)
        count += 1
        

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    print(f"## Epoch {epoch + 1}: Train Loss: {avg_train_loss} ##")
    
# try:
#     torch.cuda.memory._dump_snapshot(f"track.pickle")
# except Exception as e:
#     print(f"Failed to capture memory snapshot {e}")

# # Stop recording memory snapshot history.
# torch.cuda.memory._record_memory_history(enabled=None)


