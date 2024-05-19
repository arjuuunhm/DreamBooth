import torch
from transformers import ViTFeatureExtractor, ViTModel
from torch.nn.functional import cosine_similarity
from PIL import Image
import numpy as np

feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16')

def get_embeddings(images):
    inputs = feature_extractor(images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  

def computedino_metric(real_images, generated_images):
    real_embeddings = get_embeddings(real_images)
    generated_embeddings = get_embeddings(generated_images)

    cos_sim_matrix = cosine_similarity(real_embeddings.unsqueeze(1), generated_embeddings.unsqueeze(0), dim=-1)
    sims = cos_sim_matrix.max(dim=1).values
    return sims

if __name__ == "__main__":
    real_images = [ Image.open("/content/01.jpg")]
    # generated_images = [Image.open("/content/IMG_3206.jpeg"), Image.open("/content/IMG_5520.jpeg"), Image.open("/content/IMG_7603.jpeg"), Image.open("/content/IMG_9431.jpeg"), Image.open("/content/Screenshot 2024-05-18 at 3.48.28 PM.jpeg")]
    generated_images = [Image.open("/content/IMG_5520.jpeg")]

    scores = computedino_metric(real_images, generated_images)
    for i, score in enumerate(scores):
        print(f"DINO Metric for real image {i}: {score.item()}")