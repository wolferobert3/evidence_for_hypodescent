import torch
import dnnlib
import legacy
import PIL.Image
import numpy as np
from tqdm.notebook import tqdm
import PIL
import os
from os import listdir, mkdir
from os.path import isdir
import random

#Obtain list of directories of processed images divided by self-identification
target_folders = listdir('/stylegan2-ada-pytorch/out_source')

#Define folders of processed images divided by CFD self-identification
wm_folders = [folder for folder in target_folders if folder[4:6] == 'WM']
bm_folders = [folder for folder in target_folders if folder[4:6] == 'BM']
lm_folders = [folder for folder in target_folders if folder[4:6] == 'LM']
am_folders = [folder for folder in target_folders if folder[4:6] == 'AM']

wf_folders = [folder for folder in target_folders if folder[4:6] == 'WF']
bf_folders = [folder for folder in target_folders if folder[4:6] == 'BF']
lf_folders = [folder for folder in target_folders if folder[4:6] == 'LF']
af_folders = [folder for folder in target_folders if folder[4:6] == 'AF']

#Define pretrained network, steps of transformation, output directory
NETWORK = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
STEPS = 21
outdir = f'/stylegan2-ada-pytorch/intermediates_21/'

#Load network from pickle
network_pkl = NETWORK
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

#Divide male directories and female directories, define labels for each
all_male = [bm_folders,lm_folders,am_folders,wm_folders]
all_female = [bf_folders,lf_folders,af_folders,wf_folders]

male_dir_labels = ['bm','lm','am','wm']
female_dir_labels = ['bf','lf','af','wf']

both = [all_male,all_female]
both_labels = [male_dir_labels,female_dir_labels]

#Iterate first over male folders + labels, then female folders + labels
for n in range(len(both)):
    morph_list = both[n]
    dir_labels = both_labels[n]

	#Iterate over each source (Asian, Black, Latina/o) vs. destination (White, -1) series
    for idx in range(len(morph_list)-1):
        series_1 = morph_list[idx]
        series_2 = morph_list[-1]
		
		#Set destination directory and create it
        dir_label = f'{dir_labels[idx]}_{dir_labels[-1]}'
        series_target = f'{outdir}/{dir_label}'

        if not os.path.exists(series_target):
            os.makedirs(series_target)

		#Set maximum possible generations per individual, based on number of individuals in source series
        max_comp = int(1000/len(series_1)) + 1
        num_targets = list(range(len(series_2)))

		#Iterate over each individual in source series
        for i in range(len(series_1)):
		
			#Read in latent for source
            folder_1 = series_1[i]
            lvec1 = np.load(f'/stylegan2-ada-pytorch/out_source/{folder_1}/projected_w.npz')['w']

			#Randomly select destination individuals
            target_morphs = random.sample(num_targets,max_comp)

			#Iterate over destination individuals
            for j in target_morphs:
			
				#Read in latent
                folder_2 = series_2[j]
                lvec2 = np.load(f'/stylegan2-ada-pytorch/out_source/{folder_2}/projected_w.npz')['w']

				#Define directory path for intermediate images
                dir_path = f'{series_target}/{folder_1}_{folder_2}'
                if not isdir(dir_path):
                    mkdir(dir_path)

				#Parts of the code below are derived from the excellent tutorial by Jeff Heaton: https://github.com/jeffheaton/stylegan2-toys/blob/main/morph_video_real.ipynb
				#Define transformation parameters based on difference between images
                diff = lvec2 - lvec1
                step = diff / STEPS
                current = lvec1.copy()
                target_uint8 = np.array([1024,1024,3], dtype=np.uint8)

				#Produce a synthetic image at each step of the transformation and save it to the destination directory
                for j in range(STEPS):
                  z = torch.from_numpy(current).to(device)
                  synth_image = G.synthesis(z, noise_mode='const')
                  synth_image = (synth_image + 1) * (255/2)
                  synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

                  k = PIL.Image.fromarray(synth_image, 'RGB')
                  k.save(os.path.join(dir_path,f'intermediate_{j}.png'))
                  current = current + step