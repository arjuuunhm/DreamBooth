from diffusers import StableDiffusionPipeline
import transformers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import RandAugment
from IPython.core.debugger import set_trace
import os

import random
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from dataset_type import CustomImageDataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model to use off huggingface
model_id = "CompVis/stable-diffusion-v1-4"

# Path to directory containing images of the subject we want to use dreambooth on
dataset_path = '/home/ahm247/dreambooth/data/dataset/dog6'

# Path to our 200 photos of our prior found online. For another class generate the data
classes_path = '/home/ahm247/dreambooth/data/class-images'

#Path to checkpoint
checkpoint_path = '/home/ahm247/dreambooth/checkpoints'

# Prior and fine-tuning prompts
prior_prompt = 'A photo of a dog'
id_prompt = 'A photo of a mytoken dog'

finetuned_pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                                torch_dtype=torch.float32,
                                                use_safetensors=True,
                                                variant="fp16",
                                                safety_checker = None,
                                                requires_safety_checker = False)
finetuned_pipe.to(device)

prior_token = finetuned_pipe.tokenizer(prior_prompt, return_tensors='pt').to(device)
id_token = finetuned_pipe.tokenizer(id_prompt, return_tensors='pt').to(device)

prior_input_ids = prior_token['input_ids']
prior_attention_masks = prior_token['attention_mask']

with torch.no_grad():
    prior_encoder_hidden = finetuned_pipe.text_encoder(input_ids=prior_input_ids, attention_mask=prior_attention_masks)
    
id_input_ids = id_token['input_ids']
id_attention_masks = id_token['attention_mask']

with torch.no_grad(): 
    id_encoder_hidden = finetuned_pipe.text_encoder(input_ids=id_input_ids, attention_mask=id_attention_masks)

del prior_token
del prior_input_ids
del prior_attention_masks

del id_token
del id_input_ids
del id_attention_masks

torch.cuda.empty_cache()

# Setting up the datasets/dataloaders
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

to_pil = transforms.ToPILImage()

prior_dataset = CustomImageDataset(directory=classes_path, transform=transform)
id_dataset = CustomImageDataset(directory=dataset_path, transform=transform)
prior_dataloader = torch.utils.data.DataLoader(prior_dataset, batch_size=1, shuffle=True)
id_dataloader = torch.utils.data.DataLoader(id_dataset, batch_size=1, shuffle=True)

NUM_EPOCHS = 3
accumulation_steps = 4
finetuned_pipe.unet.train()
optimizer = optim.AdamW(finetuned_pipe.unet.parameters(), 
                        lr=5e-6,
                        betas=(0.9,0.999),
                        weight_decay=1e-2,
                        eps=1e-08)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
mse_loss = nn.MSELoss()
max_timesteps = finetuned_pipe.scheduler.num_train_timesteps

for epoch in range(NUM_EPOCHS):
    
    total_loss = 0
    num_batches = 0

    for i, prior_images in enumerate(prior_dataloader):
        prior_images = prior_images.to(torch.float32).to(device)
        id_batch = next(iter(id_dataloader))
        id_images = random.choice(id_batch).unsqueeze(0).to(torch.float32).to(device)
        
        # Add gradual noise
        noise_level = torch.rand(1, device=device)
        
        with torch.no_grad():
            prior_latent = finetuned_pipe.vae.encode(prior_images).latent_dist.sample()
            prior_latent *= 0.18215
            noisy_prior_latent = prior_latent + noise_level * torch.randn_like(prior_latent)

            id_latent = finetuned_pipe.vae.encode(id_images).latent_dist.sample()
            id_latent *= 0.18215
            noisy_id_latent = id_latent + noise_level * torch.randn_like(id_latent)

        # Sample a timestep
        timestep = torch.randint(0, max_timesteps, (1,), device=device).long()

        # Forward pass
        denoised_prior_latent = finetuned_pipe.unet(noisy_prior_latent, timestep=timestep, encoder_hidden_states=prior_encoder_hidden.last_hidden_state).sample
        denoised_id_latent = finetuned_pipe.unet(noisy_id_latent, timestep=timestep, encoder_hidden_states=id_encoder_hidden.last_hidden_state).sample

        # Calculate loss
        loss_pr = mse_loss(denoised_prior_latent, prior_latent)
        loss_id = mse_loss(denoised_id_latent, id_latent)
        loss = (loss_id + loss_pr) / accumulation_steps
        loss.backward()

        # Free up memory
        del denoised_id_latent
        del noisy_id_latent
        del id_latent
        del denoised_prior_latent
        del noisy_prior_latent
        del prior_latent
        torch.cuda.empty_cache()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

    average_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")

# Save model weights
saved_name = 'main_dog.pth'
torch.save(finetuned_pipe.unet.state_dict(), os.path.join(checkpoint_path, saved_name))

different_prompts = [
    'A photo of a mytoken dog swimming',
    'A photo of a mytoken dog in a city',
    'A photo of a mytoken dog wearing sunglasses',
    'A photo of a mytoken dog in Paris'
]
# Save image from fine-tuned model
for prompt in different_prompts: 
    final_image = finetuned_pipe(prompt).images[0]
    image_output_directory = '/home/ahm247/dreambooth/results/final_images'
    if not os.path.exists(image_output_directory):
        os.makedirs(image_output_directory)

    # Save the image in the directory
    image_path = os.path.join(image_output_directory, prompt + str(1) + '.jpg')
    final_image.save(image_path)

print('Images saved')