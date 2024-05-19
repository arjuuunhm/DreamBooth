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

# You can use one of the subjects in the provided dataset or you can provide a path to your own images. Use 3-5
dataset_path = '/home/ahm247/dreambooth/data/dataset/dog6'

# Set path to the checkpoint you just trained
checkpoint = '/home/ahm247/dreambooth/checkpoints/main_dog.pth'

# Change this prompt to whatever you want
prompts = [
    "A photo of a mytoken dog at the acropolis",
    "A photo of a mytoken dog at the beach",
    "A photo of a mytoken dog eating a sandwich",
    "A photo of a mytoken dog in a car",
    "A photo of a mytoken dog driving"
]

finetuned_pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                                torch_dtype=torch.float32,
                                                use_safetensors=True,
                                                variant="fp16",
                                                safety_checker = None,
                                                requires_safety_checker = False)

finetuned_unet_state_dict = torch.load(checkpoint)
finetuned_pipe.unet.load_state_dict(finetuned_unet_state_dict)
finetuned_pipe.to(device)

for prompt in prompts: 
    for i in range(5):
        final_image = finetuned_pipe(prompt).images[0]

        image_output_directory = '/home/ahm247/dreambooth/results/final_images'
        if not os.path.exists(image_output_directory):
            os.makedirs(image_output_directory)
        image_path = os.path.join(image_output_directory, prompt + str(i) + '.jpg')
        final_image.save(image_path)
    
print('Saved all photos')