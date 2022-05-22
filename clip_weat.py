import torch
import numpy as np
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from PIL import Image
from os import listdir, path, mkdir
from matplotlib import pyplot as plt
import random
from scipy.stats import skew, spearmanr, pearsonr, f_oneway
import seaborn as sns

#Define attribute words for WEAT test
pleasant = sorted(list(set('caress,freedom,health,love,peace,cheer,friend,heaven,loyal,pleasure,diamond,gentle,honest,lucky,rainbow,diploma,gift,honor,miracle,sunrise,family,happy,laughter,paradise,vacation'.split(','))))
unpleasant = sorted(list(set('abuse,crash,filth,murder,sickness,accident,death,grief,poison,stink,assault,disaster,hatred,pollute,tragedy,divorce,jail,poverty,ugly,cancer,kill,rotten,vomit,agony,prison'.split(','))))

#Define model
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

model.eval()

#Obtain embeddings of pleasant, unpleasant words, and normalize them
with torch.no_grad():
    pleasant_tokens = tokenizer(pleasant,return_tensors='pt',padding=True,truncation=True)
    pleasant_embeddings = model.get_text_features(**pleasant_tokens).numpy().squeeze()
    pleasant_normed = pleasant_embeddings / np.linalg.norm(pleasant_embeddings,axis=1,keepdims=True)

    unpleasant_tokens = tokenizer(unpleasant,return_tensors='pt',padding=True,truncation=True)
    unpleasant_embeddings = model.get_text_features(**unpleasant_tokens).numpy().squeeze()
    unpleasant_normed = unpleasant_embeddings / np.linalg.norm(unpleasant_embeddings,axis=1,keepdims=True)

#Vectorized SC-WEAT adapted for language + image representations
#Takes a matrix of target embeddings (v); and two matrices of attribute embeddings (B, A); assumes vectors are normalized
def veat(v,B,A):

	#Obtain associations of image with unpleasant words (B), pleasant words (A)
    B_association = B @ v.T
    A_association = A @ v.T
	
	#Concatenate associations to take standard deviation of all associations for each image vector
    joint_associations = np.concatenate((B_association,A_association),axis=0)

	#Obtain means of unpleasant associations, pleasant associations, standard deviation of all associations, for each image vector
    B_mean = np.mean(B_association,axis=0)
    A_mean = np.mean(A_association,axis=0)
    joint_std = np.std(joint_associations,axis=0,ddof=1)

	#Vector of effect sizes based on SC-WEAT formula
    veat_vector = (B_mean - A_mean) / joint_std

    return veat_vector

#Define group for which correlation of WEAT and race association will be obtained
race_1 = 'Black'

#Obtain vectors for race association, person association
with torch.no_grad():
    r1 = tokenizer([f'a photo of a {race_1} person'],return_tensors='pt')
    r1_emb = model.get_text_features(**r1).numpy().squeeze()

#Normalize race association vectors for cosine similarity
r1_normed = r1_emb / np.linalg.norm(r1_emb,keepdims=True)

#Define repository where embeddings are kept, and randomly sample 1,000 embedding matrixes corresponding to unique GAN series
source_dir = f'/home/stylegan2-ada-pytorch/clip_embeddings_21/bm_wm'
img_embs = random.sample(listdir(source_dir),1000)

#Declare lists to hold WEATs, race associations
v_list,r1_list = [],[]

#Iterate over 1,000 GAN series
for np_img in img_embs:

	#Read in 21-embedding matrix corresponding to each step of the series
    with open(path.join(source_dir,np_img),'rb') as np_reader:
        embs = np.load(np_reader)

	#Normalize vectors for cosine similarity operations
    img_normed = embs / np.linalg.norm(embs,axis=1,keepdims=True)

	#SC-WEAT associations with unpleasantness vs. pleasantness
    v = veat(img_normed,unpleasant_normed,pleasant_normed)
    v_list.append(v)

	#Cosine similarity with race label
    r1_sim = img_normed @ r1_normed
    r1_list.append(r1_sim)

#Form arrays for WEATs, race associations
v_arr = np.array(v_list)
r1_arr = np.array(r1_list)

#Obtain mean WEAT, race association at each step of the GAN series
v_means = np.mean(v_arr,axis=0)
r1_means = np.mean(r1_arr,axis=0)

#Print means
print(v_means)
print(r1_means)

#Print correlation between the mean WEATs and the mean race associations
print(pearsonr(v_means,r1_means))

#Flatten arrays
flattened_v = v_arr.flatten()
flattened_r = r1_arr.flatten()

#Obtain correlation between association with race and WEAT effect size
print(pearsonr(flattened_v,flattened_r))

