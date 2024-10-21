#VeRI Dataset
import json
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader,Sampler
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
def set_global_seed(seed=42):
    # Python built-in random
    import random
    random.seed(seed)
    
    # NumPy
    import numpy as np
    np.random.seed(seed)
    
    # PyTorch
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    
    # PyTorch Deterministic Behavior (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


encoder = OneHotEncoder(sparse_output=False)
set_global_seed(42)

# Function to load data from a JSON file
def load_data(json_file, attribute):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return [(img_name, int(attributes[attribute])) for img_name, attributes in data.items()]

# Custom Dataset class - Stage 1 
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


class VehicleDataset(Dataset):
    def __init__(self, label_to_samples, transform=None):
        self.label_to_samples = label_to_samples
        self.labels = list(label_to_samples.keys())
        self.samples = [(label, img_path) for label, img_paths in label_to_samples.items() for img_path in img_paths]
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(label)




class PKSampler(Sampler):
    def __init__(self, data_source, p=64, k=16):
        super().__init__(data_source)
        self.p = p
        self.k = k
        self.data_source = data_source

    def __iter__(self):
        # Generate batches by selecting `p` vehicle IDs and `k` images per vehicle
        pk_count = len(self) // (self.p * self.k)
        for _ in range(pk_count):
            # Randomly choose `p` unique vehicle IDs
            labels = np.random.choice(np.arange(len(self.data_source.label_to_samples.keys())), self.p, replace=False)
            
            for l in labels:
                indices = self.data_source.label_to_samples[l]
                # If less than `k` samples exist for a vehicle, allow replacement
                replace = len(indices) < self.k
                for i in np.random.choice(indices, self.k, replace=replace):
                    yield i

    def __len__(self):
        pk = self.p * self.k
        samples = ((len(self.data_source) - 1) // pk + 1) * pk
        return samples

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


'''Section 4 : Stage 2 - Data Prep'''
def get_label_to_samples():
    #train json path
    json_path = "/project/shah/shireesh/VehicleREID/train_data.json"

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create a dictionary that maps vehicleID to its list of images
    label_to_samples = {}
    base_dir = '/home/sporalas/VeRi/image_train/'
    for image_name, info in data.items():
        vehicle_id = info['vehicleID']
        image_path = os.path.join(base_dir, image_name)
        if vehicle_id not in label_to_samples:
            label_to_samples[vehicle_id] = []
        
        label_to_samples[vehicle_id].append(image_path)
        
    return label_to_samples




