import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm
from utils.network_utils import *
from networks.architectures.base_modules import *

class Encoder(nn.Module):
	def __init__(self, opt):
		super(Encoder, self).__init__()
		self.ic = opt.oc
		self.oc = opt.z_dim
		self.res_num = [64, 128, 256, 512, 512, 512]

		prev_res = self.ic
		self.blocks = []
		for res in self.res_num:
			self.blocks.append(ConvBlock(prev_res, res, 3, 2, 1, True, 'instancenorm', 'leakyrelu', False))
			prev_res = res
		self.blocks = nn.Sequential(*self.blocks)

		layer_num = len(self.res_num)
		self.final_h, self.final_w = 256 // (2**layer_num), 256 // (2**layer_num)
		self.mu = nn.Linear(self.final_h * self.final_w * prev_res, self.oc)
		self.sigma = nn.Linear(self.final_h * self.final_w * prev_res, self.oc)

	def forward(self, x):
		out = F.interpolate(x, size = (256, 256), mode = 'nearest')
		out = self.blocks(out).reshape(x.shape[0], -1)
		mu, sigma = self.mu(out), self.sigma(out)
		return mu, sigma