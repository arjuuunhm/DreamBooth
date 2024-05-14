from diffusers import DiffusionPipeline, StableDiffusionPipeline
import transformers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import RandAugment
from IPython.core.debugger import set_trace
import os

import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model to use off huggingface
model_id = "CompVis/stable-diffusion-v1-4"

# Path to directory containing images of the subject we want to use dreambooth on
dataset_path = '/home/ahm247/dreambooth/dataset/dog6'

# Path to our 200 photos of our prior found online. For another class generate the data
classes_path = '/home/ahm247/dreambooth/class-images'

# Prior and fine-tuning prompts
prior_prompt = 'A dog'
id_prompt = 'A sks dog'

# Define Dataset class for prior images
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

finetuned_pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                                torch_dtype=torch.float16,
                                                use_safetensors=True,
                                                variant="fp16")
finetuned_pipe.to(device)

prior_token = finetuned_pipe.tokenizer([prior_prompt])
id_token = finetuned_pipe.tokenizer([id_prompt])

# Setting up the datasets/dataloaders
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

prior_dataset = CustomImageDataset(directory=classes_path, transform=transform)
id_dataset = CustomImageDataset(directory=dataset_path, transform=transform)
prior_dataloader = torch.utils.data.DataLoader(prior_dataset, batch_size=1, shuffle=True)
id_dataloader = torch.utils.data.DataLoader(id_dataset, batch_size=1, shuffle=True)

NUM_EPOCHS = 10
finetuned_pipe.unet.train()
optimizer = optim.AdamW(finetuned_pipe.unet.parameters(), 
                        lr=5e-6,
                        betas=(0.9,0.999),
                        weight_decay=1e-2,
                        eps=1e-08)
mse_loss = nn.MSELoss()
print('Start of train loop')
for epoch in range(NUM_EPOCHS):

    for (prior_images, id_images) in zip(prior_dataloader, id_dataloader):
        prior_images = prior_images.to(torch.float16).to(device)
        id_images = id_images.to(torch.float16).to(device)
        print('In double for-loop')

        prior_latent = finetuned_pipe.vae.encode(prior_images).latent_dist.sample()
        print('got latent')
        prior_latent *= 0.18215
        noisy_prior_latent = prior_latent + torch.randn_like(prior_latent)

        id_latent = finetuned_pipe.vae.encode(id_images).latent_dist.sample()
        id_latent *= 0.18215
        noisy_id_latent = id_latent + torch.randn_like(id_latent).latent_dist.sample()

        denoised_prior_latent = finetuned_pipe.unet(noisy_prior_latent, timesteps=pipe.scheduler.timesteps, encoder_hidden_states=prior_token)
        denoised_id_latent = finetuned_pipe.unet(noisy_id_latent, timesteps=pipe.scheduler.timesteps, encoder_hidden_states=id_token) 

        # Forward pass
        optimizer.zero_grad()

        # Calculate loss
        loss_id = mse_loss(id_token, input_images)
        loss_pr = mse_loss(noise_image, prior_images)
        loss = loss_id + loss_pr

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save model weights
saved_name = 'main_dog.pth'
torch.save(finetuned_pipe.unet.state_dict(), os.path.join('.','checkpoints', saved_name))