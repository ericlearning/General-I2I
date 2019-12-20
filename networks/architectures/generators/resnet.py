import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from utils.network_utils import *
from networks.architectures.base_modules import *

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

class ResNet_G(nn.Module):
	def __init__(self, opt):
		super(ResNet_G, self).__init__()
		self.ic = opt.ic
		self.oc = opt.oc
		self.h, self.w = opt.height, opt.width
		self.sz = max(self.h, self.w)
		self.z_dim = opt.z_dim
		use_sn = opt.use_sn_G
		norm_type = opt.norm_type_G

		self.dims = {'16' : 2, '32' : 3, '64' : 4, '128' : 5, '256' : 6, '512' : 7}
		num_res = self.dims[str(self.sz)]
		self.conv = nn.Sequential(*[
			nn.ReflectionPad2d(3),
			ConvBlock(self.ic + self.z_dim, 64, 7, 1, 0, use_sn, None, None, True)
		])
		self.conv_block1 = ConvBlock(64, 128, 3, 2, 1, use_sn, norm_type, 'leakyrelu', False)
		self.conv_block2 = ConvBlock(128, 256, 3, 2, 1, use_sn, norm_type, 'leakyrelu', False)

		self.resblocks = nn.Sequential(*[ResBlock(256, 256, use_sn, norm_type, 'relu') for _ in range(num_res)])

		self.deconv_block1 = ConvTransposedBlock(256, 128, 3, 2, 1, 1, use_sn, norm_type, 'leakyrelu', False)
		self.deconv_block2 = ConvTransposedBlock(128, 64, 3, 2, 1, 1, use_sn, norm_type, 'leakyrelu', False)
		self.deconv = nn.Sequential(*[
			nn.ReflectionPad2d(3),
			ConvBlock(64, self.oc, 7, 1, 0, use_sn, None, None, True)
		])

		self.tanh = nn.Tanh()

	def forward(self, x, z):
		if(self.z_dim > 0): x = expand_and_concat(x, z)
		out = self.conv(x)
		out = self.conv_block1(out)
		out = self.conv_block2(out)
		out = self.resblocks(out)
		out = self.deconv_block1(out)
		out = self.deconv_block2(out)
		out = self.deconv(out)
		out = self.tanh(out)
		return out