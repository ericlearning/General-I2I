import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.transforms as transforms
import torch.nn.functional as F
from loss.gan_loss import *
from PIL import Image

def get_norm(norm_type, size):
	if(norm_type == 'batchnorm'):
		return nn.BatchNorm2d(size)
	elif(norm_type == 'instancenorm'):
		return nn.InstanceNorm2d(size)
	elif(norm_type == None):
		return Nothing()

def get_activation(activation_type):
	if(activation_type == 'relu'):
		return nn.ReLU()
	elif(activation_type == 'leakyrelu'):
		return nn.LeakyReLU(0.2)
	elif(activation_type == 'elu'):
		return nn.ELU()
	elif(activation_type == 'selu'):
		return nn.SELU()
	elif(activation_type == 'prelu'):
		return nn.PReLU()
	elif(activation_type == 'tanh'):
		return nn.Tanh()
	elif(activation_type == None):
		return Nothing()

def get_require_type(loss_type):
	if(loss_type == 'SGAN' or loss_type == 'LSGAN' or loss_type == 'HINGEGAN' or loss_type == 'WGAN'):
		require_type = 0
	elif(loss_type == 'RASGAN' or loss_type == 'RALSGAN' or loss_type == 'RAHINGEGAN'):
		require_type = 1
	else:
		require_type = -1
	return require_type

def get_gan_loss(loss_type):
	loss_dict = {'SGAN':SGAN, 'LSGAN':LSGAN, 'HINGEGAN':HINGEGAN, 'WGAN':WGAN, 'RASGAN':RASGAN, \
				'RALSGAN':RALSGAN, 'RAHINGEGAN':RAHINGEGAN}
	require_type = get_require_type(loss_type)

	if(require_type == 0):
		loss = loss_dict[loss_type]()
	elif(require_type == 1):
		loss = loss_dict[loss_type]()
	else:
		loss = None

	return loss

def get_transformations(opt, data_type):
	h, w, ic, oc = opt.height, opt.width, opt.ic, opt.oc
	if(data_type == 'image'):
		dt = {
			'input' : transforms.Compose([
				transforms.Resize((h, w)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			]),
			'target' : transforms.Compose([
				transforms.Resize((h, w)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])
		}
	elif(data_type == 'label'):
		dt = {
			'input' : transforms.Compose([
				transforms.Resize((h, w), interpolation = Image.NEAREST)
			]),
			'target' : transforms.Compose([
				transforms.Resize((h, w)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])
		}
	return dt

def get_stage(cur_epoch, total_epoch, network_type, freeze_epoch):
	if(network_type == 'MultiStep'):
		p1, p2 = total_epoch // 2, total_epoch // 2 + freeze_epoch
		if(cur_epoch < p1):
			return 'global'
		elif(p1 <= cur_epoch <= p2):
			return 'local'
		elif(p2 < cur_epoch):
			return 'finetune'

	else:
		return 'normal'

def gradient_penalty(self, D, x, real_image, fake_image):
	bs = real_image.size(0)
	alpha = torch.FloatTensor(bs, 1, 1, 1).uniform_(0, 1).expand(real_image.size()).to(self.device)
	interpolation = alpha * real_image + (1 - alpha) * fake_image

	c_xi = D(x, interpolation)
	gradients = autograd.grad(c_xi, interpolation, torch.ones(c_xi.size()).to(self.device),
							  create_graph = True, retain_graph = True, only_inputs = True)[0]
	gradients = gradients.view(bs, -1)
	penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
	return penalty
	
def generate_noise(bs, nz):
	if(nz == 0):
		return None
	noise = torch.randn(bs, nz, 1, 1)
	return noise

def resize(x, scale):
	out = x
	if(scale > 1):
		size = (x.shape[2] // scale, x.shape[3] // scale)
		out = F.interpolate(x, size = size, mode = 'nearest')
	return out

def expand_and_concat(x1, x2):
	return torch.cat([x1, x2.expand(-1, -1, x1.shape[2], x1.shape[3])], 1)

class Nothing(nn.Module):
	def __init__(self):
		super(Nothing, self).__init__()
		
	def forward(self, x):
		return x

class UpSample(nn.Module):
	def __init__(self):
		super(UpSample, self).__init__()

	def forward(self, x):
		return F.interpolate(x, None, 2, 'nearest')

def xavier_init(m):
	if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
		nn.init.xavier_normal_(m.weight)

def normal_init(m, v = 0.02):
	if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
		m.weight.data.normal_(0.0, v)
		if(m.bias is not None):
			m.bias.data.zero_()

def reparametarize(mu, log_sigma):
	sigma = (log_sigma * 0.5).exp()
	z = torch.randn_like(sigma) * sigma + mu
	z = z.unsqueeze(2).unsqueeze(3)
	return z

def get_normalization(norm_type, device):
	if(norm_type == 'imagenet'):
		mean, std = torch.Tensor([0.485, 0.456, 0.406]), torch.Tensor([0.229, 0.224, 0.225])
	mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
	std = std.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
	return mean, std

def normalize(x, norm):
	mean, std = norm
	norm_x = (x + 1) / 2.0
	norm_x = (norm_x - mean) / std
	return norm_x

class CustomDataParallel(nn.Module):
	def __init__(self, m):
		super(CustomDataParallel, self).__init__()
		self.m = nn.DataParallel(m)

	def forward(self, *x):
		return self.m(*x)

	def __getattr__(self, attr):
		try:
			return super().__getattr__(attr)
		except:
			return getattr(self.m.module, attr)

def split(x, num):
	if(num > 1):
		bs = x.shape[0]
		return x.split(bs // num, dim = 0)
	else:
		return [x]

