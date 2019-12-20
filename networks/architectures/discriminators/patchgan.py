import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.network_utils import *
from networks.architectures.base_modules import ConvBlock

class PatchGan(nn.Module):
	def __init__(self, opt):
		super(PatchGan, self).__init__()
		ic = opt.ic + opt.oc
		use_sn = opt.use_sn_D
		norm_type = opt.norm_type_D
		use_sigmoid = opt.use_sigmoid

		if(use_sigmoid):
			self.act = nn.Sigmoid()
		else:
			self.act = Nothing()

		self.convs = nn.ModuleList([
			ConvBlock(ic, 64, 4, 2, 1, use_sn, None, 'leakyrelu'),
			ConvBlock(64, 128, 4, 2, 1, use_sn, norm_type, 'leakyrelu'),
			ConvBlock(128, 256, 4, 2, 1, use_sn, norm_type, 'leakyrelu'),
			ConvBlock(256, 512, 4, 1, 1, use_sn, norm_type, 'leakyrelu'),
			ConvBlock(512, 1, 4, 1, 1, use_sn, None, None),
			self.act
		])
		
	def forward(self, x1, x2):
		outs = []
		out = torch.cat([x1, x2], 1)

		for conv in self.convs:
			out = conv(out)
			outs.append(out)

		return outs

class MultiScale_PatchGan(nn.Module):
	def __init__(self, opt):
		super(MultiScale_PatchGan, self).__init__()
		scale_num = opt.D_scale_num
		self.discriminators = nn.ModuleList([PatchGan(opt) for _ in range(scale_num)])

	def forward(self, x1, x2):
		outs, features = [], []
		for i, D in enumerate(self.discriminators):
			out = D(x1, x2)
			outs.append(out)
			x1, x2 = resize(x1, 2), resize(x2, 2)

		return outs

