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
from loss import PriorPreservationLoss

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


frozen_pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                torch_dtype=torch.float16,
                                                use_safetensors=True,
                                                variant="fp16")

finetuned_pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                                torch_dtype=torch.float16,
                                                use_safetensors=True,
                                                variant="fp16")

frozen_pipe.to(device)
finetuned_pipe.to(device)

# Data Loader
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# We will figure out the specifics for this

# Fine-tuning

