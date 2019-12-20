import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from utils.network_utils import *
from networks.architectures.base_modules import *

class UNet_G(nn.Module):
	def __init__(self, opt):
		super(UNet_G, self).__init__()
		self.ic = opt.ic
		self.oc = opt.oc
		self.h, self.w = opt.height, opt.width
		self.sz = max(self.h, self.w)
		self.z_dim = opt.z_dim
		use_sn = opt.use_sn_G
		norm_type = opt.norm_type_G

		self.dims = {
			'16' : [64, 128, 256, 512],
			'32' : [64, 128, 256, 512, 512],
			'64' : [64, 128, 256, 512, 512, 512],
			'128' : [64, 128, 256, 512, 512, 512, 512],
			'256' : [64, 128, 256, 512, 512, 512, 512, 512],
			'512' : [64, 128, 256, 512, 512, 512, 512, 512, 512]
		}
		self.cur_dim = self.dims[str(self.sz)]
		self.num_convs = len(self.cur_dim)

		self.leaky_relu = nn.LeakyReLU(0.2)
		self.relu = nn.ReLU()

		self.enc_convs = nn.ModuleList([])
		cur_block_ic = self.ic + self.z_dim
		for i, dim in enumerate(self.cur_dim):
			if(i == 0 or i == len(self.cur_dim) - 1):
				self.enc_convs.append(ConvBlock(cur_block_ic, dim, 4, 2, 1, use_sn, None, None, True))
			else:
				self.enc_convs.append(ConvBlock(cur_block_ic, dim, 4, 2, 1, use_sn, norm_type, None, False))
			cur_block_ic = dim

		self.dec_convs = nn.ModuleList([])
		cur_block_ic = self.cur_dim[-1]
		for i, dim in enumerate(list(reversed(self.cur_dim))[1:] + [self.oc]):
			if(i == 0):
				self.dec_convs.append(ConvTransposedBlock(cur_block_ic, dim, 4, 2, 1, 0, use_sn, None, None, True))
			elif(i == len(self.cur_dim) - 1):
				self.dec_convs.append(ConvTransposedBlock(cur_block_ic*2, self.oc, 4, 2, 1, 0, use_sn, None, None, True))
			else:
				self.dec_convs.append(ConvTransposedBlock(cur_block_ic*2, dim, 4, 2, 1, 0, use_sn, norm_type, None, False))
			cur_block_ic = dim

		self.tanh = nn.Tanh()
	
	def forward(self, x, z):
		ens = []
		if(self.z_dim > 0): x = expand_and_concat(x, z)
		for i, cur_enc in enumerate(self.enc_convs):
			if(i == 0):
				out = cur_enc(x)
			else:
				out = cur_enc(self.leaky_relu(out))
			ens.append(out)

		for i, cur_dec in enumerate(self.dec_convs):
			cur_enc = ens[self.num_convs - 1 - i]
			if(i == 0):
				out = cur_dec(self.relu(cur_enc))
			else:
				out = cur_dec(self.relu(torch.cat([out, cur_enc], 1)))
		del ens
		out = self.tanh(out)
		return out
