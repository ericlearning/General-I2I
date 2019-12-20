import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm
from utils.network_utils import *
from networks.architectures.base_modules import *

def ref_pad(conv, pad):
	return nn.Sequential(nn.ReflectionPad2d(pad), conv)

class ResBlock(nn.Module):
	def __init__(self, ic, oc, use_sn, norm_type, act_type = 'relu'):
		super(ResBlock, self).__init__()
		self.reflection_pad1 = nn.ReflectionPad2d(1)
		self.reflection_pad2 = nn.ReflectionPad2d(1)

		self.conv1 = ConvBlock(ic, oc, 3, 1, 0, use_sn, norm_type, 'relu', False)
		self.conv2 = ConvBlock(oc, oc, 3, 1, 0, use_sn, norm_type, 'relu', False)

	def forward(self, x):
		out = self.reflection_pad1(x)
		out = self.conv1(out)
		out = self.reflection_pad2(out)
		out = self.conv2(out)
		out = out + x
		return out

class Pix2PixHD_G(nn.Module):
	def __init__(self, opt):
		super(Pix2PixHD_G, self).__init__()
		self.ic = opt.ic
		self.oc = opt.oc
		self.z_dim = opt.z_dim
		use_sn = opt.use_sn_G
		norm_type = opt.norm_type_G

		self.global_G = Global(self.ic + self.z_dim, self.oc, use_sn, norm_type)
		self.local_G1 = Local(self.ic + self.z_dim, self.oc, use_sn, norm_type, 0)
		self.local_G2 = Local(self.ic + self.z_dim, self.oc, use_sn, norm_type, 1)

		self.cur_stage = -1

	def freeze(self, models, choices):
		for model, choice in zip(models, choices):
			for child in model.children():
				for param in child.parameters():
					param.requires_grad = not choice

	def forward(self, x, z, stage = 'normal'):
		x_half = resize(x, 2)
		if(self.z_dim > 0):
			x = expand_and_concat(x, z)
			x_half = expand_and_concat(x_half, z)
		if(stage == 'global'):
			if(self.cur_stage != stage):
				self.freeze((self.local_G1, self.local_G2, self.global_G), (True, True, False))
				self.cur_stage = stage
			out = self.global_G(x, use_last_conv = True)

		elif(stage == 'local'):
			if(self.cur_stage != stage):
				self.freeze((self.local_G1, self.local_G2, self.global_G), (False, False, True))
				self.cur_stage = stage
			out1 = self.local_G1(x)
			out2 = self.global_G(x_half, use_last_conv = False)
			out = self.local_G2(out1 + out2)

		elif(stage == 'finetune'):
			if(self.cur_stage != stage):
				self.freeze((self.local_G1, self.local_G2, self.global_G), (False, False, False))
				self.cur_stage = stage
			out1 = self.local_G1(x)
			out2 = self.global_G(x_half, use_last_conv = False)
			out = self.local_G2(out1 + out2)

		elif(stage == 'normal'):
			out1 = self.local_G1(x)
			out2 = self.global_G(x_half, use_last_conv = False)
			out = self.local_G2(out1 + out2)

		return out

class Global(nn.Module):
	def __init__(self, ic, oc, use_sn, norm_type):
		super(Global, self).__init__()
		self.conv1 = ref_pad(ConvBlock(ic, 64, 7, 1, 0, use_sn, norm_type, 'relu', False), 3)
		self.blocks = nn.Sequential(
			ref_pad(ConvBlock(64, 128, 3, 2, 0, use_sn, norm_type, 'relu', False), 1),
			ref_pad(ConvBlock(128, 256, 3, 2, 0, use_sn, norm_type, 'relu', False), 1),
			ref_pad(ConvBlock(256, 512, 3, 2, 0, use_sn, norm_type, 'relu', False), 1),
			ref_pad(ConvBlock(512, 1024, 3, 2, 0, use_sn, norm_type, 'relu', False), 1)
		)
		self.res = [ResBlock(1024, 1024, use_sn, norm_type, 'relu') for _ in range(9)]
		self.res = nn.Sequential(*self.res)

		self.blocks2 = nn.Sequential(
			ConvTransposedBlock(1024, 512, 3, 2, 1, 1, use_sn, norm_type, 'relu', False),
			ConvTransposedBlock(512, 256, 3, 2, 1, 1, use_sn, norm_type, 'relu', False),
			ConvTransposedBlock(256, 128, 3, 2, 1, 1, use_sn, norm_type, 'relu', False),
			ConvTransposedBlock(128, 64, 3, 2, 1, 1, use_sn, norm_type, 'relu', False)
		)
		self.conv2 = ref_pad(ConvBlock(64, oc, 7, 1, 0, use_sn, None, None, True), 3)
		self.tanh = nn.Tanh()

	def forward(self, x, use_last_conv = True):
		out = self.conv1(x)
		out = self.blocks(out)
		out = self.res(out)
		out = self.blocks2(out)
		if(use_last_conv):
			out = self.conv2(out)
			out = self.tanh(out)
		return out

class Local(nn.Module):
	def __init__(self, ic, oc, use_sn, norm_type, part):
		super(Local, self).__init__()
		if(part == 0):
			self.module = nn.Sequential(*[
				ref_pad(ConvBlock(ic, 32, 7, 1, 0, use_sn, norm_type, 'relu', False), 3),
				ref_pad(ConvBlock(32, 64, 3, 2, 0, use_sn, norm_type, 'relu', False), 1)
			])
		elif(part == 1):
			self.module = nn.Sequential(*[
				ResBlock(64, 64, use_sn, norm_type, 'relu'),
				ResBlock(64, 64, use_sn, norm_type, 'relu'),
				ResBlock(64, 64, use_sn, norm_type, 'relu'),
				ConvTransposedBlock(64, 32, 3, 2, 1, 1, use_sn, norm_type, 'relu', False),
				ref_pad(ConvBlock(32, oc, 7, 1, 0, use_sn, None, None, True), 3),
				nn.Tanh()
			])

	def forward(self, x):
		out = self.module(x)
		return out
