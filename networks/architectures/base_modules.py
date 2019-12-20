import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
import torchvision.models as model
from utils.network_utils import get_norm, get_activation

class ConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad, use_sn, norm_type, act_type, bias = False):
		super(ConvBlock, self).__init__()
		self.use_sn = use_sn
		self.norm_type = norm_type

		self.conv = nn.Conv2d(ni, no, ks, stride, pad, bias = bias)
		if(self.use_sn):
			self.conv = SpectralNorm(self.conv)

		self.norm = get_norm(norm_type, no)
		self.act = get_activation(act_type)

	def forward(self, x):
		out = x
		out = self.conv(out)
		out = self.norm(out)
		out = self.act(out)
		return out

class ConvTransposedBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad, out_pad, use_sn, norm_type, act_type, bias = False):
		super(ConvTransposedBlock, self).__init__()
		self.use_sn = use_sn
		self.norm_type = norm_type

		self.conv = nn.ConvTranspose2d(ni, no, ks, stride, pad, output_padding = out_pad, bias = bias)
		if(self.use_sn):
			self.conv = SpectralNorm(self.conv)

		self.norm = get_norm(norm_type, no)
		self.act = get_activation(act_type)

	def forward(self, x):
		out = x
		out = self.conv(out)
		out = self.norm(out)
		out = self.act(out)
		return out

class VGG19(nn.Module):
	def __init__(self):
		super(VGG19, self).__init__()
		self.f = model.vgg19(pretrained = True).features
		self.split = [0, 2, 7, 12, 21, 30]

	def forward(self, x):
		outs = []
		out = x
		for i in range(len(self.split) - 1):
			out = self.f[self.split[i]:self.split[i+1]](out)
			outs.append(out)
		return outs

class BatchNorm2D_noparam(nn.Module):
	def __init__(self, eps = 1e-8):
		super(BatchNorm2D_noparam, self).__init__()
		self.eps = eps

	def forward(self, x):
		bs, c, h, w = x.shape
		mean = torch.mean(x, (0, 2, 3), keepdim = True)
		var = torch.var(x, (0, 2, 3), keepdim = True)
		out = ((x - mean) / (var + self.eps))
		return out
