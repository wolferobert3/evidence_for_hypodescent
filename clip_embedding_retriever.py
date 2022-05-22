import torch
import numpy as np
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from PIL import Image
from os import listdir, path, mkdir

#Define model
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

model.eval()

#Set embedding destination path, image source path
clip_path = f'/stylegan2-ada-pytorch/clip_embeddings_21'
image_directories = [f'/stylegan2-ada-pytorch/intermediates_21/{i}' for i in listdir(f'/stylegan2-ada-pytorch/intermediates_21')]

#Create directories for each pair of groups
for d in image_directories:
    if not path.isdir(f'{clip_path}/{d[-5:]}'):
        mkdir(f'{clip_path}/{d[-5:]}')

#Iterate through image source directories
for top_dir in image_directories:

	#Find subdirectories for each transformation
    sub_dirs = [path.join(top_dir,sub_dir) for sub_dir in listdir(top_dir)]
    current_clip_path = f'{clip_path}/{top_dir[-5:]}'

	#Iterate through subdirectories for each GAN transformation
    for directory in sub_dirs:
	
		#Read in each image from the subdirectory in order of transformation index
        img_list = []
        imgs = [f'intermediate_{num}.png' for num in range(0,21)]

		#Obtain embedding for each image in the subdirectory
        for image in imgs:

            intermediate_image = Image.open(path.join(directory,image))

            img = processor(images=intermediate_image,return_tensors='pt')

            with torch.no_grad():
                img_emb = model.get_image_features(**img).numpy().squeeze()

            img_list.append(img_emb)

		#Form ordered array of embeddings based on transformation index
        img_arr = np.array(img_list)
		
		#Save ordered array to destination directory - name of current subdirectory corresponds to pair of images transformed
        save_ = directory.split('/')[-1]
        with open(f'{current_clip_path}/{save_}.npy','wb') as np_write:
            np.save(np_write,img_arr)