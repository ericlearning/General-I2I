import torch
import torch.nn as nn
import numpy as np

def get_label(bs):
	label_r = torch.full((bs, ), 1)
	label_f = torch.full((bs, ), 0)
	return label_r, label_f

class SGAN(nn.Module):
	def __init__(self):
		super(SGAN, self).__init__()
		self.criterion = nn.BCELoss()

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs)
		return self.criterion(c_xr, label_r) + self.criterion(c_xf, label_f)

	def g_loss(self, c_xf):
		bs = c_xf.shape[0]
		label_r, _ = get_label(bs)
		return self.criterion(c_xf, label_r)

class LSGAN(nn.Module):
	def __init__(self):
		super(LSGAN, self).__init__()

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs)
		return torch.mean((c_xr - label_r) ** 2) + torch.mean((c_xf - label_f) ** 2)

	def g_loss(self, c_xf):
		bs = c_xf.shape[0]
		label_r, _ = get_label(bs)
		return torch.mean((c_xf - label_r) ** 2)

class HINGEGAN(nn.Module):
	def __init__(self):
		super(HINGEGAN, self).__init__()

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		return torch.mean(torch.nn.ReLU()(1-c_xr)) + torch.mean(torch.nn.ReLU()(1+c_xf))

	def g_loss(self, c_xf):
		return -torch.mean(c_xf)

class WGAN(nn.Module):
	def __init__(self):
		super(WGAN, self).__init__()

	def d_loss(self, c_xr, c_xf):
		return -torch.mean(c_xr) + torch.mean(c_xf)

	def g_loss(self, c_xf):
		return -torch.mean(c_xf)

class RASGAN(nn.Module):
	def __init__(self):
		super(RASGAN, self).__init__()
		self.criterion = nn.BCEWithLogitsLoss()

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs)
		return (self.criterion(c_xr - torch.mean(c_xf), label_r) + self.criterion(c_xf - torch.mean(c_xr), label_f)) / 2.0

	def g_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs)
		return (self.criterion(c_xr - torch.mean(c_xf), label_f) + self.criterion(c_xf - torch.mean(c_xr), label_r)) / 2.0

class RALSGAN(nn.Module):
	def __init__(self):
		super(RALSGAN, self).__init__()

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, _ = get_label(bs)
		return (torch.mean((c_xr - torch.mean(c_xf) - label_r)**2) + torch.mean((c_xf - torch.mean(c_xr) + label_r)**2)) / 2.0

	def g_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, _ = get_label(bs)
		return (torch.mean((c_xf - torch.mean(c_xr) - label_r)**2) + torch.mean((c_xr - torch.mean(c_xf) + label_r)**2)) / 2.0

class RAHINGEGAN(nn.Module):
	def __init__(self):
		super(RAHINGEGAN, self).__init__()

	def d_loss(self, c_xr, c_xf):
		return (torch.mean(torch.nn.ReLU()(1-(c_xr-torch.mean(c_xf)))) + torch.mean(torch.nn.ReLU()(1+(c_xf-torch.mean(c_xr))))) / 2.0

	def g_loss(self, c_xr, c_xf):
		return (torch.mean(torch.nn.ReLU()(1-(c_xf-torch.mean(c_xr)))) + torch.mean(torch.nn.ReLU()(1+(c_xr-torch.mean(c_xf))))) / 2.0
