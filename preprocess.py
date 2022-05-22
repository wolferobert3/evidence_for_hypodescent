from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
import dlib
from os import path, listdir

#The functions in this code are derived from the excellent tutorial by Jeff Heaton: https://github.com/jeffheaton/stylegan2-toys/blob/main/morph_video_real.ipynb
#Load in DLIB detector, predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

#Function for finding the eyes of an individual using DLIB
def find_eyes(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	if len(rects) == 0:
		print(image_)
	elif len(rects) > 1:
		print(image_)

	shape = predictor(gray, rects[0])
	features = []

	for i in range(0, 5):
		features.append((i, (shape.part(i).x, shape.part(i).y)))

	return (int(features[3][1][0] + features[2][1][0]) // 2, \
		int(features[3][1][1] + features[2][1][1]) // 2), \
		(int(features[1][1][0] + features[0][1][0]) // 2, \
		int(features[1][1][1] + features[0][1][1]) // 2)

#StyleGAN cropping function to ensure images align well with FFHQ dataset
def crop_stylegan(img):
	left_eye, right_eye = find_eyes(img)
	d = abs(right_eye[0] - left_eye[0])
	z = 255/d
	ar = img.shape[0]/img.shape[1]
	w = img.shape[1] * z
	img2 = cv2.resize(img, (int(w), int(w*ar)))
	bordersize = 1024
	img3 = cv2.copyMakeBorder(
		img2,
		top=bordersize,
		bottom=bordersize,
		left=bordersize,
		right=bordersize,
		borderType=cv2.BORDER_REPLICATE)

	left_eye2, right_eye2 = find_eyes(img3)

	crop1 = left_eye2[0] - 385
	crop0 = left_eye2[1] - 490
	return img3[crop0:crop0+1024,crop1:crop1+1024]

#Define source and destination directories
dest_dir = f'/home/stylegan2-ada-pytorch/cfd/cfd3/divided_images/N_processed/'
image_dir = f'/home/stylegan2-ada-pytorch/cfd/cfd3/divided_images/N/'

#Get list of images to crop such that they're similar to FFHQ framing
images_to_process = listdir(image_dir)

#Iterate over all images
for image_ in images_to_process:

	#Read in each image
    image_source = cv2.imread(path.join(image_dir,image_))
    if image_source is None:
        raise ValueError("Source image not found")

	#Crop the image and write it to destination
    cropped_source = crop_stylegan(image_source)
    img = cv2.cvtColor(cropped_source, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path.join(dest_dir,image_), cropped_source)
