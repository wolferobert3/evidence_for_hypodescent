import torch
import numpy as np
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from PIL import Image
from os import listdir, path, mkdir
from matplotlib import pyplot as plt
import random
from scipy.stats import skew, spearmanr, pearsonr
import pandas as pd

#Define model
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

model.eval()

#Define groups under comparison
race_1 = 'Black'
race_2 = 'White'

#Obtain projected text embedding for each race label, person label
with torch.no_grad():
    r1 = tokenizer([f'a photo of a {race_1} person'],return_tensors='pt')
    r2 = tokenizer([f'a photo of a {race_2} person'],return_tensors='pt')
    mu = tokenizer(['a photo of a multiracial person'],return_tensors='pt')
    p = tokenizer(['a photo of a person'],return_tensors='pt')

    r1_emb = model.get_text_features(**r1).numpy().squeeze()
    r2_emb = model.get_text_features(**r2).numpy().squeeze()
    mu_emb = model.get_text_features(**mu).numpy().squeeze()
    p_emb = model.get_text_features(**p).numpy().squeeze()

#Normalize vectors
r1_normed = r1_emb / np.linalg.norm(r1_emb,keepdims=True)
r2_normed = r2_emb / np.linalg.norm(r2_emb,keepdims=True)
mu_normed = mu_emb / np.linalg.norm(mu_emb,keepdims=True)
p_normed = p_emb / np.linalg.norm(p_emb,keepdims=True)

#Define repository where embeddings are kept, and randomly sample 1,000 embedding matrixes corresponding to unique GAN series
source_dir = f'/home/stylegan2-ada-pytorch/clip_embeddings_21/bf_wf'
img_embs = random.sample(listdir(source_dir),1000)

#Declare lists to hold label associations
r1_list = []
r2_list = []
mu_list = []
p_list = []

#Declare list to hold number of times race 1 association is higher at each step of the GAN series
counts = [0 for _ in range(21)]

#Iterate over 1,000 GAN series
for np_img in img_embs:

	#Read in 21-embedding matrix corresponding to each step of the series
    with open(path.join(source_dir,np_img),'rb') as np_reader:
        embs = np.load(np_reader)

	#Normalize vectors for cosine similarity operations
    img_normed = embs / np.linalg.norm(embs,axis=1,keepdims=True)

	#Cosine similarity with each race label, person label
    cos_sims_r1 = img_normed @ r1_normed
    cos_sims_r2 = img_normed @ r2_normed
    cos_sims_mu = img_normed @ mu_normed
    cos_sims_p = img_normed @ p_normed

	#Iterate over cosine similarities and log whether association with race 1 (i.e., Black) is higher than with race 2 (White)
    for i in range(len(cos_sims_r1)):
        if cos_sims_r1[i] > cos_sims_r2[i]:
            counts[i] += 1

	#Add cosine similarities to label association lists
    r1_list.append(cos_sims_r1)
    r2_list.append(cos_sims_r2)
    mu_list.append(cos_sims_mu)
    p_list.append(cos_sims_p)

#Print raw counts out of 1,000 for which race 1 association is higher
print('counts')

#Print percentage of associations for which race 1 is higher
pct_counts = [i/10 for i in counts]
print(pct_counts)

#Print skew of the distribution
dist = skew(counts)
print(dist)

#Form arrays from association lists
r1_arr = np.array(r1_list)
r2_arr = np.array(r2_list)
mu_arr = np.array(mu_list)
p_arr = np.array(p_list)

#Obtain and print means for each label association
r1_means = np.mean(r1_arr,axis=0)
r2_means = np.mean(r2_arr,axis=0)
mu_means = np.mean(mu_arr,axis=0)
p_means = np.mean(p_arr,axis=0)

print(f'Means of {race_1} associations across full GAN series')
print(r1_means)
print(f'Means of {race_2} associations across full GAN series')
print(r2_means)
print(f'Means of multiracial associations across full GAN series')
print(mu_means)
print(f'Means of person associations across full GAN series')
print(p_means)

#Flatten arrays for calculation of standard deviation, correlation of all race associations with person
flat_r1 = r1_arr.flatten()
flat_r2 = r2_arr.flatten()
flat_p = p_arr.flatten()

#Standard Deviations across full GAN series
print(f'Standard Deviation of {race_1} associations across full GAN series')
print(np.std(flat_r1))
print(f'Standard Deviation of {race_2} associations across full GAN series')
print(np.std(flat_r2))

#Correlations of race associations with person associations
print(f'Correlation of {race_1} with person')
print(pearsonr(flat_r1,flat_p))
print(f'Correlation of {race_2} with person')
print(pearsonr(flat_r2,flat_p))
