import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from utils.network_utils import *
from networks.architectures.base_modules import *

def conv3x3(ic, oc, use_sn = True, use_bias = False):
	reflection = nn.ReflectionPad2d(1)
	conv = ConvBlock(ic, oc, 3, 1, 0, use_sn, None, None, use_bias)
	return nn.Sequential(reflection, conv)

class SPADE(nn.Module):
	def __init__(self, ic_1, ic_2, use_sn = True):
		super(SPADE, self).__init__()
		self.bn = BatchNorm2D_noparam()
		self.conv = conv3x3(ic_2, 128, use_sn, True)
		self.gamma_conv = conv3x3(128, ic_1, use_sn, True)
		self.beta_conv = conv3x3(128, ic_1, use_sn, True)
		self.relu = nn.ReLU()

	def forward(self, x, con):
		normalized = self.bn(x)
		r_con = F.interpolate(con, size = (x.shape[2], x.shape[3]), mode = 'nearest')
		r_con = self.relu(self.conv(r_con))
		gamma = self.gamma_conv(r_con)
		beta = self.beta_conv(r_con)

		out = gamma * normalized + beta
		return out

class SPADE_ResBlk(nn.Module):
	def __init__(self, ic, oc, channel_c, use_sn = True):
		super(SPADE_ResBlk, self).__init__()
		self.spade_1 = SPADE(ic, channel_c, use_sn)
		self.spade_2 = SPADE(ic, channel_c, use_sn)

		self.conv_1 = conv3x3(ic, ic, use_sn)
		self.conv_2 = conv3x3(ic, oc, use_sn)

		self.learned_skip = (ic != oc)
		if(self.learned_skip):
			self.spade_skip = SPADE(ic, channel_c, use_sn)
			self.conv_skip = conv3x3(ic, oc, use_sn)

		self.relu = nn.LeakyReLU(0.2)

	def forward(self, x, con):
		out = self.spade_1(x, con)
		out = self.relu(out)
		out = self.conv_1(out)
		out = self.spade_2(out, con)
		out = self.relu(out)
		out = self.conv_2(out)

		if(self.learned_skip):
			skip_out = self.spade_skip(x, con)
			skip_out = self.relu(skip_out)
			skip_out = self.conv_skip(skip_out)
		else:
			skip_out = x

		return out + skip_out

class SPADE_G(nn.Module):
	def __init__(self, opt):
		super(SPADE_G, self).__init__()
		self.ic = opt.ic
		self.oc = opt.oc
		self.h, self.w = opt.height, opt.width
		self.network_mode = opt.network_mode
		self.z_dim = opt.z_dim
		use_sn = opt.use_sn_G

		if(self.network_mode == 'normal'):
			upsample_num, prev_res = 5, 64
		elif(self.network_mode == 'more'):
			upsample_num, prev_res = 6, 64
		elif(self.network_mode == 'most'):
			upsample_num, prev_res = 7, 32

		self.b1 = SPADE_ResBlk(1024, 1024, self.ic, use_sn)
		self.b2 = SPADE_ResBlk(1024, 1024, self.ic, use_sn)
		self.b3 = SPADE_ResBlk(1024, 1024, self.ic, use_sn)
		self.b4 = SPADE_ResBlk(1024, 512, self.ic, use_sn)
		self.b5 = SPADE_ResBlk(512, 256, self.ic, use_sn)
		self.b6 = SPADE_ResBlk(256, 128, self.ic, use_sn)
		self.b7 = SPADE_ResBlk(128, 64, self.ic, use_sn)
		self.b8 = SPADE_ResBlk(64, 32, self.ic, use_sn)

		self.conv = conv3x3(prev_res, self.oc, use_sn, True)
		self.relu = nn.LeakyReLU(0.2)
		self.tanh = nn.Tanh()
		self.upsample = UpSample()

		self.init_sz = (self.h // (2**upsample_num), self.w // (2**upsample_num))

		if(self.z_dim > 0):
			self.linear = nn.Linear(self.z_dim, self.init_sz[0] * self.init_sz[1] * 1024)
		else:
			self.linear = conv3x3(self.ic, 1024, False, True)

	def forward(self, x, z):
		if(self.z_dim > 0):
			out = self.linear(z.view(-1, self.z_dim))
			out = out.view(-1, 1024, self.init_sz[0], self.init_sz[1])
		else:
			out = F.interpolate(x, size = (self.init_sz[0], self.init_sz[1]), mode = 'nearest')
			out = self.linear(out)
		
		out = self.b1(out, x)
		out = self.upsample(out)
		out = self.b2(out, x)

		if(self.network_mode == 'more' or self.network_mode == 'most'):
			out = self.upsample(out)

		out = self.b3(out, x)
		out = self.upsample(out)
		out = self.b4(out, x)
		out = self.upsample(out)
		out = self.b5(out, x)
		out = self.upsample(out)
		out = self.b6(out, x)
		out = self.upsample(out)
		out = self.b7(out, x)

		if(self.network_mode == 'most'):
			out = self.upsample(out)
			out = self.b8(out, x)
			
		out = self.relu(out)
		out = self.conv(out)
		out = self.tanh(out)
		
		return out
