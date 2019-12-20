import os
import torch
import random
import pickle
import numpy as np
from utils.network_utils import get_transformations
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class Dataset():
	def __init__(self, opt):
		self.type = opt.data_type
		self.train_dir = [opt.trn_src_pth, opt.trn_trg_pth]
		self.val_dir = [opt.val_src_pth, opt.val_trg_pth]
		self.label_num = opt.ic
		self.label_pth = opt.label_pth
		self.dt = get_transformations(opt, self.type)
		self.num_workers = opt.num_workers

	def get_loader(self, bs):
		input_transform = self.dt['input']
		target_transform = self.dt['target']

		if(self.type == 'image'):
			trn_dataset = Pix2Pix_Dataset(self.train_dir[0], self.train_dir[1], input_transform, target_transform)
			val_dataset = Pix2Pix_Dataset(self.val_dir[0], self.val_dir[1], input_transform, target_transform)
		elif(self.type == 'label'):
			trn_dataset = Label2Pix_Dataset(self.train_dir[0], self.train_dir[1], input_transform, target_transform, self.label_pth, self.label_num)
			val_dataset = Label2Pix_Dataset(self.val_dir[0], self.val_dir[1], input_transform, target_transform, self.label_pth, self.label_num)

		trn_loader = DataLoader(trn_dataset, batch_size = bs, shuffle = True, num_workers = self.num_workers)
		val_loader = list(DataLoader(val_dataset, batch_size = 3, shuffle = False, num_workers = self.num_workers))[0]

		returns = (trn_loader, val_loader)
		return returns

class Pix2Pix_Dataset():
	def __init__(self, input_dir, target_dir, input_transform, target_transform):
		self.input_dir = input_dir
		self.target_dir = target_dir
		self.input_transform = input_transform
		self.target_transform = target_transform

		self.image_name_list = []
		for file in os.listdir(input_dir):
			if(file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.bmp')):
				self.image_name_list.append(file)

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):
		input_img = Image.open(os.path.join(self.input_dir, self.image_name_list[idx]))
		target_img = Image.open(os.path.join(self.target_dir, self.image_name_list[idx]))

		input_img = self.input_transform(input_img)
		target_img = self.target_transform(target_img)

		sample = (input_img, target_img)
		return sample

class Label2Pix_Dataset():
	def __init__(self, input_dir, target_dir, input_transform, target_transform, labels_file, label_num):
		self.input_dir = input_dir
		self.target_dir = target_dir
		self.input_transform = input_transform
		self.target_transform = target_transform
		self.labels_file = labels_file
		self.label_num = label_num

		self.image_name_list = []
		for file in os.listdir(input_dir):
			if(file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.bmp')):
				self.image_name_list.append(file)

		self.labels = self.get_label_dict()

	def get_label_dict(self):
		if(os.path.exists(self.labels_file)):
			with open(self.labels_file, 'rb') as f:
				labels = pickle.load(f)
		else:
			labels = {}
			cnt = 0
			for file in self.image_name_list:
				input_fn = os.path.join(self.input_dir, file)
				input_img = np.array(Image.open(input_fn))
				for n in np.unique(input_img.reshape(-1, input_img.shape[2]), axis = 0):
					pixel = tuple(n)
					if(labels.get(pixel) is None):
						labels[pixel] = cnt
						cnt += 1
				if(cnt >= self.label_num - 1):
					break
			with open(self.labels_file, 'wb') as f:
				pickle.dump(labels, f)

		return labels

	def one_hot_label(self, img):
		img = np.array(img)
		label = np.zeros((img.shape[0], img.shape[1], self.label_num), dtype = np.float32)
		label[:, :, -1] = 1
		for rgb, cnt in self.labels.items():
			mask = (rgb == img).all(2)
			label[mask, cnt] = 1
			label[mask, -1] = 0
		label = torch.from_numpy(label).permute(2, 0, 1)
		return label

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):
		input_img = Image.open(os.path.join(self.input_dir, self.image_name_list[idx]))
		target_img = Image.open(os.path.join(self.target_dir, self.image_name_list[idx]))

		input_img = self.input_transform(input_img)
		input_img = self.one_hot_label(input_img)
		target_img = self.target_transform(target_img)

		sample = (input_img, target_img)
		return sample