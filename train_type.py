import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models


from data_preprocessing import Load_Type_Train_Data, set_global_seed
from resnet_model1 import ResNet
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


'''Important'''
set_global_seed(42)


# Load the data
try:
    train_loader, val_loader, test_loader = Load_Type_Train_Data()
    print("\n### Data loaded successfully ###")
except:
    print("\n### Error occured at loading the dataset ###")
    
# Initialize the model (ResNet50 for 9 classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n### Device {device} is used . ###")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 9) 
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # weight decay for regularization

# Training parameters
num_epochs = 10
best_val_accuracy = 0.0
best_model_path = 'best_resnet50_type.pth'

# Variables to store metrics for plotting
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = torch.argmax(labels, dim=1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    train_accuracy = 100 * correct_train / total_train
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation loop
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0  # Initialize validation loss
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = torch.argmax(labels, dim=1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculate validation loss
            val_loss += loss.item()  # Accumulate validation loss

            _, predicted = outputs.max(1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= len(val_loader)  # Average validation loss
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Testing using the best model
model.load_state_dict(torch.load(best_model_path))
model.eval()

correct_test = 0
total_test = 0
test_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = torch.argmax(labels, dim=1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = outputs.max(1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

        # Collect all predictions and labels for F1, recall, precision, and confusion matrix
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


test_accuracy = 100 * correct_test / total_test
test_loss /= len(test_loader)

# Compute Precision, Recall, F1 Score
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# Print the computed metrics
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix and save as image
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')  # Save confusion matrix as an image
plt.close()

# Plot training & validation loss
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')  # Save loss plot as an image
plt.close()

# Plot training & validation accuracy
plt.figure()
plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('accuracy_plot.png')  # Save accuracy plot as an image
plt.close()

