import torch
import os
from PIL import Image

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory 
        self.transform = transform
        self.image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
    
        if self.transform:
            image = self.transform(image)

        return image