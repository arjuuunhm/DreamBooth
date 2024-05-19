# Implementation of Google's Dreambooth: Fine Tuning Text-to-Image Diffusion Models for Subject Driven Generation

3.1 – Introduction
Fine Tuning Text-to-Image Diffusion Models for Subject Driven Generation (Dreambooth) was published at CVPR 2023 by Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman. They are all researchers at Google and Nataniel Ruiz is also a Professor at Boston University. The motivation behind this project is that large text-to-image models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts.

The goal of this approach is to take a pre-trained text-to-image model (In the paper it is Imagen and in our implementation it is a Stable Diffusion model) and finetune it so that it learns to bind a certain identifier token [V] with the subject of the 3-5 images. An example of this is if I have a specific photo of my dog. Dreambooth seeks to leverage the pre-trained model’s prior understanding of what a dog is and entangle it with the embedding of my dog’s unique identifier [V]. 

The main contributions that the paper introduces are the following: 
1. Rare-identifier token 
The authors found that using common words messed with the model’s prior understanding of said words, and using gibberish like “xxxyyy55ttt” lead to artifacts in images rendered by the model. They proposed using words that have low frequency in the diffusion model’s language model.
2. Prior-preservation Loss
The fine-tuning process introduces language drift. This leads to the model losing its semantic understanding for the class and overfitting. The authors introduce a prior-preservation loss as a regularization term.
The authors introduce the use of a frozen model for this. The sample images of their class (Ex. a dog) from the frozen model and attempt to recover a noised version of those samples in the fine tuned model. The fine tuned model also attempts to recover noised samples of [V] class. This encourages the fine-tuned model to learn to bind the [V] token with the subject but also not forget its understanding of the class itself. 
3. Proposed the DINO metric which measures subject fidelity of generated images

3.2 – Chosen Result
The results we attempted to replicate were the DreamBooth (Stable Diffusion) DINO results which was introduced by the group. The DINO metric is the average pairwise cosine similarity between the ViT-S/16 DINO embeddings of generated and real images.

Below is a table where you can see the results from the paper. Pay attention to the Stable Diffusion results since that is the one we aimed to replicate. This is because Imagen is not a public model.

It is worth noting that this DINO metric is not perfect because our subject will be in different orientations and scenarios. As you can see in the diagram under real images DINO score is 0.774. We can see DreamBooth with Stable Diffusion has a score of 0.668. 

