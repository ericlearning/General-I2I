import torch
import torch.nn as nn
from utils.network_utils import *
from networks.network import Image2Image_Network

class Image2Image_Trainer():
	def __init__(self, opt, iter_num):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.network = Image2Image_Network(opt, self.device).to(self.device)
		if(opt.multigpu):
			self.network.D = CustomDataParallel(self.network.D)
			self.network.G = CustomDataParallel(self.network.G)
		self.optD, self.optG = self.network.initialize_optimizers(iter_num)
		self.grad_acc = opt.grad_acc

	def preprocess_input(self, inputs, stage):
		x, y = inputs[0].to(self.device), inputs[1].to(self.device)
		if(self.network.netG_type == 'MultiStep' and stage == 'global'):
			x, y = resize(x, 2), resize(y, 2)
		x, y = split(x, self.grad_acc), split(y, self.grad_acc)
		return x, y

	def step(self, inputs, stage):
		x, y = self.preprocess_input(inputs, stage)
		errD = self.D_one_step((x, y), stage)
		errG = self.G_one_step((x, y), stage)
		return errD, errG

	def D_one_step(self, inputs, stage):
		x, y = inputs
		self.network.D.zero_grad()
		for x_, y_ in zip(x, y):
			errD = self.network((x_, y_), 'discriminator', stage) / self.grad_acc
			errD.backward()
		self.optD.step()
		return errD

	def G_one_step(self, inputs, stage):
		x, y = inputs
		self.network.G.zero_grad()
		for x_, y_ in zip(x, y):
			errG = self.network((x_, y_), 'generator', stage) / self.grad_acc
			errG.backward()
		self.optG.step()
		return errG

	def save(self, filename):
		state = {
			'netD' : self.network.D.state_dict,
			'netG' : self.network.G.state_dict,
			'optD' : self.optD.state_dict,
			'optG' : self.optG.state_dict
		}
		torch.save(state, filename)

	def load(self, filename):
		state = torch.load(filename)
		self.network.D.load_state_dict(state['netD'])
		self.network.G.load_state_dict(state['netG'])
		self.optD.load_state_dict(state['optD'])
		self.optG.load_state_dict(state['optG'])