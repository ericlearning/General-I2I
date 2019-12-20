import torch
import torch.nn as nn
from utils.network_utils import *
from networks.architectures.base_modules import VGG19

class GanLoss(nn.Module):
	def __init__(self, opt):
		super(GanLoss, self).__init__()
		gan_loss_type = opt.gan_loss_type

		self.opt = opt
		self.loss = get_gan_loss(gan_loss_type)
		self.require_type = get_require_type(gan_loss_type)

	def d_loss(self, inputs):
		c_xr, c_xf = inputs

		if(self.require_type == 0 or self.require_type == 1):
			errD = self.loss.d_loss(c_xr, c_xf)
		return errD

	def g_loss(self, inputs):
		c_xr, c_xf = inputs

		if(self.require_type == 0):
			errG = self.loss.g_loss(c_xf)
		elif(self.require_type == 1):
			errG = self.loss.g_loss(c_xr, c_xf)
		return errG

	def forward(self, inputs, mode):
		loss = 0
		c_xr, c_xf = inputs
		scale_num = self.opt.D_scale_num
		if(mode == 'dis'):
			for i in range(scale_num):
				loss += self.d_loss((c_xr[i][-1], c_xf[i][-1]))
		elif(mode == 'gen'):
			for i in range(scale_num):
				loss += self.g_loss((c_xr[i][-1], c_xf[i][-1]))

		return loss / scale_num

class RecLoss(nn.Module):
	def __init__(self, opt):
		super(RecLoss, self).__init__()
		self.opt = opt
		self.weight = opt.rec_weight
		self.L1 = nn.L1Loss()

	def forward(self, inputs, mode):
		rec_loss = 0
		if(self.weight > 0):
			if(mode == 'pixel'):
				y, fake_y = inputs
				rec_loss = self.weight * self.L1(y, fake_y)
			elif(mode == 'feature'):
				c_xr, c_xf = inputs
				scale_num = self.opt.D_scale_num
				for i in range(scale_num):
					s = 0
					for j in range(len(c_xr[i])-1):
						s += self.weight * self.L1(c_xr[i][j], c_xf[i][j])
					rec_loss += s
				rec_loss /= scale_num

		return rec_loss

class VGGLoss(nn.Module):
	def __init__(self, opt, norm):
		super(VGGLoss, self).__init__()
		self.weight = opt.vgg_weight
		self.weight_per_module = [1/32, 1/16, 1/8, 1/4, 1]
		self.vgg19 = VGG19()
		self.L1 = nn.L1Loss()
		self.norm = norm

	def forward(self, inputs):
		vgg_loss = 0
		if(self.weight > 0):
			y, fake_y = inputs
			y, fake_y = normalize(y, self.norm), normalize(fake_y, self.norm)
			fs_a, fs_b = self.vgg19(y), self.vgg19(fake_y)
			for f_a, f_b, w in zip(fs_a, fs_b, self.weight_per_module):
				vgg_loss += self.L1(f_a, f_b) * w * self.weight

		return vgg_loss

class DSLoss(nn.Module):
	def __init__(self, opt):
		super(DSLoss, self).__init__()
		self.weight = opt.ds_weight
		self.l1 = nn.L1Loss(reduction = 'none')
		self.z_dim = opt.z_dim

	def forward(self, inputs):
		fake_y1, fake_y2, noise1, noise2 = inputs
		fake_y_diff = self.l1(fake_y1, fake_y2).sum(1).sum(1).sum(1) / (fake_y1.shape[1]*fake_y1.shape[2]*fake_y1.shape[3])
		noise_diff = self.l1(noise1, noise2).sum(1).sum(1).sum(1) / self.z_dim
		ds_loss = -torch.mean(fake_y_diff / (noise_diff+1e-5)) * self.weight
		return ds_loss

class KLLoss(nn.Module):
	def __init__(self, opt):
		super(KLLoss, self).__init__()
		self.weight = opt.kl_weight

	def forward(self, inputs):
		kl_loss = 0
		if(self.weight > 0):
			mu, log_sigma = inputs
			loss = 0.5 * torch.sum(log_sigma.exp() + mu**2 - 1 - log_sigma) * self.weight
		return loss


