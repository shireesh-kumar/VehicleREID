#VeRI Dataset
import json
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import numpy as np
import cv2
import torch


''' Session 1 : To store specific vehicle details on a JSON file ... '''
# from lxml import etree
# import json

# # Parse the XML file with lxml
# with open('E:\\Datasets\\VeRi\\test_label.xml', 'r', encoding='gb2312') as file:
#     tree = etree.parse(file)

# root = tree.getroot()

# # Create a dictionary to store the data
# data = {}
# number_of_records = 0
# # Iterate over each "Item" element in the XML
# for item in root.findall('Items/Item'):  # Adjusted path to reflect the XML structure
#     image_name = item.get('imageName')
#     vehicleID = item.get('vehicleID')
#     cameraID = item.get('cameraID')
#     colorID = item.get('colorID')
#     typeID = item.get('typeID')
#     number_of_records += 1

#     # Store the data in a dictionary for easy lookup
#     data[image_name] = {
#         "vehicleID": vehicleID,
#         "cameraID": cameraID,
#         "colorID": colorID,
#         "typeID": typeID
#     }

# print(f"Number of records : {number_of_records}")
# # Save the dictionary to a JSON file
# with open('test_data.json', 'w') as json_file:
#     json.dump(data, json_file, indent=4)


'''Utility Methods and Classes'''

encoder = OneHotEncoder(sparse_output=False)

# Function to load data from a JSON file
def load_data(json_file, attribute):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return [(img_name, int(attributes[attribute])) for img_name, attributes in data.items()]

# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, data, img_dir, encoder, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
        self.encoder  = encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)  # Create full image path
        image = Image.open(img_path).convert('RGB')  # Load image

        if self.transform:
            image = self.transform(image)  # Apply transformations if any

        # One-hot encode the typeID (label)
        type_one_hot = encoder.transform([[label - 1]]).flatten()  # One-hot encode the label
        return image, torch.tensor(type_one_hot, dtype=torch.float32)


''' Session 2 : Load the dataset for stage 1 : Classification - Type '''

def Load_Type_Train_Data():

    # Define image transformations (optional)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
    ])     
    
    # Load training data
    train_data = load_data('/home/sporalas/VehicleREID/train_data.json','typeID')

    # Split the training data into training and validation sets
    train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=42)

    # Load testing data
    test_data = load_data('/home/sporalas/VehicleREID/test_data.json','typeID')                                                       
    
    # Create datasets
    img_dir1 = '/home/sporalas/VeRi/image_train'  # Specify the directory containing  training and validation images
    img_dir2 = '/home/sporalas/VeRi/image_test'  # Specify the directory containing  testing images
    encoder.fit(np.arange(9).reshape(-1, 1))
    train_dataset = ImageDataset(train_set, img_dir1, encoder, transform)
    val_dataset = ImageDataset(val_set, img_dir1, encoder, transform)
    test_dataset = ImageDataset(test_data, img_dir2, encoder, transform)

    # Create data loaders
    batch_size = 32  # Define batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n## Type attribute dataset training ##")
    print(f"Dataset batch size : {batch_size}")
    print(f"Number of batches in training set: {len(train_loader)}")
    print(f"Number of batches in Validation set: {len(val_loader)}")
    print(f"Number of batches in Test set : {len(test_loader)}\n")
    
    return train_loader, val_loader, test_loader



'''Session 3 : Load the dataset for stage 1 : Classification - Color '''

def Load_Color_Train_Data():
    
    # Custom transform to convert image to HSV color space
    class ConvertToHSV:
        def __call__(self, img):
            # Convert PIL image to a NumPy array
            img_np = np.array(img)
            # Convert RGB to HSV using OpenCV
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            # Convert back to PIL image for compatibility with torchvision transforms
            return Image.fromarray(img_hsv)
        
    # Define image transformations (optional)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images
            #ConvertToHSV(),  # Convert to HSV color space
            transforms.ToTensor(),  # Convert to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
    ])                                                            
    
    # Load training data
    train_data = load_data('/home/sporalas/VehicleREID/train_data.json','colorID')

    # Split the training data into training and validation sets
    train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=42)

    # Load testing data
    test_data = load_data('/home/sporalas/VehicleREID/test_data.json','colorID')                                                       

    
    # Create datasets
    img_dir1 = '/home/sporalas/VeRi/image_train'  # Specify the directory containing  training and validation images
    img_dir2 = '/home/sporalas/VeRi/image_test'  # Specify the directory containing  testing images
    encoder.fit(np.arange(10).reshape(-1, 1))
    train_dataset = ImageDataset(train_set, img_dir1,encoder,transform)
    val_dataset = ImageDataset(val_set, img_dir1,encoder, transform)
    test_dataset = ImageDataset(test_data, img_dir2,encoder, transform)

    # Create data loaders
    batch_size = 32  # Define batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n## Type attribute dataset training ##")
    print(f"Dataset batch size : {batch_size}")
    print(f"Number of batches in training set: {len(train_loader)}")
    print(f"Number of batches in Validation set: {len(val_loader)}")
    print(f"Number of batches in Test set : {len(test_loader)}\n")
    
    return train_loader, val_loader, test_loader

Load_Type_Train_Data()