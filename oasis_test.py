from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import numpy as np
import pandas as pd
import torch
from PIL import Image
from os import listdir
from scipy.stats import norm, pearsonr

#Initialize CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def SC_WEAT(w, A, B, permutations):
    w_normed = w / np.linalg.norm(w)
    A_normed = A / np.linalg.norm(A,axis=-1,keepdims=True)
    B_normed = B / np.linalg.norm(B,axis=-1,keepdims=True)

    A_associations = w_normed @ A_normed.T
    B_associations = w_normed @ B_normed.T
    joint_associations = np.concatenate((A_associations,B_associations),axis=-1)

    test_statistic = np.mean(A_associations) - np.mean(B_associations)
    effect_size = test_statistic / np.std(joint_associations,ddof=1)

    midpoint = len(A)
    sample_distribution = np.array([np.random.permutation(joint_associations) for _ in range(permutations)])
    sample_associations = np.mean(sample_distribution[:,:midpoint],axis=1) - np.mean(sample_distribution[:,midpoint:],axis=1)
    p_value = 1 - norm.cdf(test_statistic,np.mean(sample_associations),np.std(sample_associations,ddof=1))

    return effect_size, p_value

pleasant = sorted(list(set('caress,freedom,health,love,peace,cheer,friend,heaven,loyal,pleasure,diamond,gentle,honest,lucky,rainbow,diploma,gift,honor,miracle,sunrise,family,happy,laughter,paradise,vacation'.split(','))))
unpleasant = sorted(list(set('abuse,crash,filth,murder,sickness,accident,death,grief,poison,stink,assault,disaster,hatred,pollute,tragedy,divorce,jail,poverty,ugly,cancer,kill,rotten,vomit,agony,prison'.split(','))))


pleasant_embs, unpleasant_embs = [],[]

for word in pleasant:
    with torch.no_grad():
        text = tokenizer(word,return_tensors='pt')
        text_emb = model.get_text_features(**text).numpy().squeeze()
    pleasant_embs.append(text_emb)

for word in unpleasant:
    with torch.no_grad():
        text = tokenizer(word,return_tensors='pt')
        text_emb = model.get_text_features(**text).numpy().squeeze()
    unpleasant_embs.append(text_emb)

pleasant_arr = np.array(pleasant_embs)
unpleasant_arr = np.array(unpleasant_embs)

SOURCE = f'D:\\OASIS\\OASIS\\images\\'

oasis_norms = pd.read_csv(f'D:\\OASIS\\OASIS\\oasis.csv',index_col='Theme')

val_norm = oasis_norms['Valence_mean'].tolist()

imgs = oasis_norms.index.tolist()
embs = []
valence = []

for idx,img in enumerate(imgs):
    print(f'{idx} {img}')
    current_img = Image.open(f'{SOURCE}{img}.jpg').convert('RGB')
    with torch.no_grad():
        input = processor(images=current_img,return_tensors='pt')
        img_emb = model.get_image_features(**input).numpy().squeeze()
    embs.append(img_emb)
    es,p = SC_WEAT(img_emb,pleasant_arr,unpleasant_arr,10)
    valence.append(es)

print(pearsonr(valence,val_norm))