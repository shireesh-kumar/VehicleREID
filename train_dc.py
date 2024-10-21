import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms

from data_preprocessing import VehicleDataset, get_label_to_samples, PKSampler, set_global_seed

from loss import trihard_loss
from feature_extractor import CustomResNet34
from dc_module import DCModule

'''Important'''
set_global_seed(42)

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_to_samples = get_label_to_samples()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 (width, height)
    transforms.ToTensor()           # Convert the image to a tensor
])

# Create dataset
dataset = VehicleDataset(label_to_samples, transform=transform)
print(f"Len of the dataset = {len(dataset)}")
# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% training
val_size = len(dataset) - train_size  # 20% validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) 

print(len(train_dataset))
print((len(val_dataset)))

# Create DataLoader for both train and validation sets
p, k = 64, 16
train_sampler = PKSampler(train_dataset, p=p, k=k)
val_sampler = PKSampler(val_dataset, p=p, k=k)

train_loader = DataLoader(train_dataset, batch_size=p * k, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=p * k, sampler=val_sampler)

# Create an instance of the DCModule
dc_module = DCModule(window_size=3, step_size=2).to(device)   # Adjust parameters as needed


# Model definition and optimizer
model = CustomResNet34(pretrained=True).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
 
# Training loop setup
num_epochs = 2
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    
    for batch in train_loader:
        images, labels = batch
        # Move images and labels to the device
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)  # Forward pass to get embeddings
        
        # Compute triplet loss for each unique vehicle ID
        triplet_loss = 0
        triplet_loss += trihard_loss(embeddings, labels, margin=1.0)

        # Backpropagation and optimization
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()
 
        total_train_loss += triplet_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            # Move images and labels to the device
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)
            
            triplet_loss = 0
            triplet_loss += trihard_loss(embeddings, labels, margin=1.0)    
            total_val_loss += triplet_loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

# Plot and save the loss graph
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plot to a file without displaying it
plt.savefig("triplet_loss_plot.png")
