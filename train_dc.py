import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms

from data_preprocessing import  get_label_to_samples, set_global_seed, split_label_to_samples
from sampler import VehicleDataset, RandomIdentitySampler

from loss import trihard_loss
from feature_extractor import CustomResNet18
from dc_module import DCModule
import torch.nn as nn
from torchvision.models import  resnet18, ResNet18_Weights


'''Important'''
set_global_seed(42)

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"### Using device: {device} ###")
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")


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

print("### Below are the train and val dataset lengths ###")
print(f"Train : {len(train_dataset)}")
print(f"Val   : {len(val_dataset)}")
            
# Create DataLoader for both train and validation sets
p, k = 2, 16

print(f"Sampler valeus : P = {p} and K = {k}")
train_sampler = RandomIdentitySampler(train_dataset, batch_size=p*k, num_instances=k)
val_sampler = RandomIdentitySampler(val_dataset, batch_size=p*k, num_instances=k)

train_loader = DataLoader(train_dataset, batch_size=p * k, sampler=train_sampler,num_workers=4,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=p * k, sampler=val_sampler,num_workers=4,pin_memory=True)

print(f"Train Loader {len(train_loader)}")
print(f"Train Sampler = {len(train_sampler)}")

for i in train_loader:
    print(f"Batch individual size = {i[0].shape}")
    # print(f"Label Set Dimentions = {i.shape}")
    # print(f"Image Set Dimentions{b.shape}")
    break 

# Create an instance of the DCModule
dc_module = DCModule(window_size=3, step_size=2).to(device)   # Adjust parameters as needed


# Model definition and optimizer
weights =ResNet18_Weights.IMAGENET1K_V1
model = CustomResNet18(weights=weights).to(device)

# Wrap the model for multi-GPU usage
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)  # Move model to GPU

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# Mixed Precision Training Setup
scaler = torch.amp.GradScaler()  # Initialize the gradient scaler

# Training loop setup
num_epochs = 1
train_losses = []
val_losses = []

print("Initiating Training ...")
for epoch in range(num_epochs):
    print("Training Started...")
    # Training phase
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        count += 1
        images = batch[0]  # The first element is the images tensor
        labels = batch[1]  # The second element is the tuple of labels
        
        # Move images and labels to the device
        images = images.to(device)
        labels = labels.to(device) 
    
        with torch.amp.autocast(device_type='cuda'):
            print(f"Model is on device: {next(model.parameters()).device}")
            embeddings = model(images)  # Forward pass to get embeddings
            #Compute triplet loss for each unique vehicle ID
            triplet_loss = trihard_loss(embeddings, labels,dc_module, margin=1.0)
        
        print("Loss computation completed for a batch")
        # Backpropagation and optimization
        optimizer.zero_grad()
        scaler.scale(triplet_loss).backward()   # Scale the loss for backpropagation
        scaler.step(optimizer)                  # Update the parameters
        scaler.update()                         # Update the scaler

        total_train_loss += triplet_loss.item()
        
        
        # Delete the tensors if no longer needed
        del images, labels, embeddings, triplet_loss
        torch.cuda.empty_cache()
        


    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss}")




