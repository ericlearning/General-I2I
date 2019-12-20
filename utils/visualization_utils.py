import os, cv2
import pickle
import torch
import torch.nn as nn
import numpy as np
from utils.network_utils import resize

def get_display_samples(samples, n_x, n_y):
	h = samples[0].shape[0]
	w = samples[0].shape[1]
	nc = samples[0].shape[2]
	display = np.zeros((h*n_y, w*n_x, nc))
	for i in range(n_y):
		for j in range(n_x):
			cur_sample = cv2.cvtColor(samples[i*n_x+j]*255.0, cv2.COLOR_BGR2RGB)
			display[i*h:(i+1)*h, j*w:(j+1)*w, :] = cur_sample
	return display.astype(np.uint8)

def label2rgb(label_img, labels):
	label_img = label_img.argmax(axis = 0)
	channel_num = len(list(labels.keys())[0])
	img = np.zeros((label_img.shape[0], label_img.shape[1], channel_num))
	for rgb, cnt in labels.items():
		img[label_img == cnt, :] = rgb
	return img.transpose(2, 0, 1).astype(np.float32)

def get_sample_images_list(trainer, val_data, z, stage, label_path, data_type):
	device = trainer.device
	netG_type = trainer.network.netG_type

	if(data_type == 'label'):
		with open(label_path, 'rb') as f:
			labels = pickle.load(f)

	if(z is not None):
		z = z.to(device)
		z = torch.cat([z[0].unsqueeze(0)] * 3 + [z[1].unsqueeze(0)] * 3 + [z[2].unsqueeze(0)] * 3, 0)
		sample_input_images = val_data[0].repeat(3, 1, 1, 1)
		sample_output_images = val_data[1].repeat(3, 1, 1, 1)
	else:
		sample_input_images = val_data[0]
		sample_output_images = val_data[1]

	if(netG_type == 'MultiStep' and stage == 'global'):
		sample_input_images = resize(sample_input_images, 2)
		sample_output_images = resize(sample_output_images, 2)

	sample_fake_images, _ = trainer.network.generate(sample_input_images.to(device), None, z, 'eval', stage)
	sample_fake_images = sample_fake_images.detach().cpu().numpy()
	sample_input_images = sample_input_images.numpy()
	sample_output_images = sample_output_images.numpy()
	
	sample_input_images_list = []
	sample_output_images_list = []
	sample_fake_images_list = []
	sample_images_list = []

	r = 3 if(z is None) else 9
	for j in range(r):
		cur_img_fake = (sample_fake_images[j] + 1) / 2.0
		cur_img_output = (sample_output_images[j] + 1) / 2.0
		if(data_type == 'image'):
			cur_img_input = (sample_input_images[j] + 1) / 2.0
		elif(data_type == 'label'):
			cur_img_input = label2rgb(sample_input_images[j], labels) / 255.0
		sample_fake_images_list.append(cur_img_fake.transpose(1, 2, 0))
		sample_input_images_list.append(cur_img_input.transpose(1, 2, 0))
		sample_output_images_list.append(cur_img_output.transpose(1, 2, 0))

	sample_images_list.extend(sample_input_images_list)
	sample_images_list.extend(sample_fake_images_list)
	sample_images_list.extend(sample_output_images_list)

	return sample_images_list
