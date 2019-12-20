import torch
import torch.nn as nn
import torch.optim as optim
from utils.network_utils import *
from scheduler.scheduler import LinearDecay
from networks.architectures.generators.unet import *
from networks.architectures.generators.resnet import *
from networks.architectures.generators.multistep import *
from networks.architectures.generators.spade import *
from networks.architectures.discriminators.patchgan import *
from networks.architectures.encoder.encoder import *
from loss.loss import *

class Image2Image_Network(nn.Module):
	def __init__(self, opt, device):
		super(Image2Image_Network, self).__init__()
		self.opt = opt
		self.device = device
		self.rec_type = opt.rec_type
		self.netG_type = opt.net_G_type
		self.netD_type = opt.net_D_type
		self.z_dim = opt.z_dim
		self.init_type = opt.weight_init_type
		self.use_encoder = opt.use_encoder
		self.use_ds_loss = self.z_dim > 0 and self.opt.ds_weight > 0

		if(self.init_type == 'normal'):
			init = normal_init
		elif(self.init_type == 'xavier'):
			init = xavier_init

		if(self.netG_type == 'ResNet'):
			self.G = ResNet_G(opt)
		elif(self.netG_type == 'UNet'):
			self.G = UNet_G(opt)
		elif(self.netG_type == 'MultiStep'):
			self.G = Pix2PixHD_G(opt)
		elif(self.netG_type == 'SPADE'):
			self.G = SPADE_G(opt)

		if(self.netD_type == 'PatchGan'):
			self.D = MultiScale_PatchGan(opt)

		if(self.use_encoder):
			self.E = Encoder(opt)
			self.E.apply(init)

		self.G.apply(init)
		self.D.apply(init)

		vgg_input_norm = get_normalization('imagenet', self.device)
		self.GanLoss = GanLoss(opt)
		self.VGGLoss = VGGLoss(opt, vgg_input_norm)
		self.RecLoss = RecLoss(opt)
		self.DSLoss = DSLoss(opt)
		self.KLLoss = KLLoss(opt)

	def initialize_optimizers(self, iter_num):
		use_ttur = self.opt.use_ttur
		lr, beta1, beta2 = self.opt.lr, self.opt.beta1, self.opt.beta2
		G_params = list(self.G.parameters())
		D_params = list(self.D.parameters())
		if(self.use_encoder):
			G_params += list(self.E.parameters())

		if(use_ttur):
			optD = optim.Adam(D_params, lr = lr*2.0, betas = (beta1, beta2))
			optG = optim.Adam(G_params, lr = lr/2.0, betas = (beta1, beta2))
		else:
			optD = optim.Adam(D_params, lr = lr, betas = (beta1, beta2))
			optG = optim.Adam(G_params, lr = lr, betas = (beta1, beta2))
		optD, optG = LinearDecay(self.opt, optD, iter_num), LinearDecay(self.opt, optG, iter_num)

		return optD, optG


	def forward(self, inputs, mode, stage = 'normal'):
		if(mode == 'discriminator'):
			x, y = inputs
			if(self.use_encoder): var = self.E(y)
			else: var = None
			fake_y, z = self.generate(x, var, stage = stage)
			c_xr = self.D(x, y)
			c_xf = self.D(x, fake_y.detach())

			errD = self.GanLoss((c_xr, c_xf), 'dis')
			return errD

		elif(mode == 'generator'):
			x, y = inputs
			if(self.use_encoder): var = self.E(y)
			else: var = None
			fake_y, z = self.generate(x, var, stage = stage)
			c_xr = self.D(x, y)
			c_xf = self.D(x, fake_y)

			errG_gan = self.GanLoss((c_xr, c_xf), 'gen')
			errG_vgg = self.VGGLoss((y, fake_y))

			if(self.rec_type == 'pixel'):
				errG_rec = self.RecLoss((y, fake_y), self.rec_type)
			elif(self.rec_type == 'feature'):
				errG_rec = self.RecLoss((c_xr, c_xf), self.rec_type)

			if(self.use_ds_loss):
				fake_y_2, z2 = self.generate(x, stage = stage)
				errG_ds = self.DSLoss((fake_y, fake_y_2, z, z2))
			else:
				errG_ds = 0

			if(self.use_encoder):
				errG_kl = self.KLLoss(var)
			else:
				errG_kl = 0

			errG = errG_gan + errG_vgg + errG_rec + errG_ds + errG_kl
			return errG
			

	def generate(self, x, var = None, z = None, mode = 'train', stage = 'normal'):
		if(self.z_dim > 0):
			if(z is None):
				if(not self.use_encoder):
					z = generate_noise(x.shape[0], self.z_dim)
				else:
					mu, log_sigma = var
					z = reparametarize(mu, log_sigma)
				z = z.to(self.device)
		else:
			z = None

		if(mode == 'train'):
			self.G.train()
			if(self.netG_type == 'MultiStep'):
				generated = self.G(x, z, stage = stage)
			else:
				generated = self.G(x, z)
				
		elif(mode == 'eval'):
			self.G.eval()
			with torch.no_grad():
				if(self.netG_type == 'MultiStep'):
					generated = self.G(x, z, stage = stage)
				else:
					generated = self.G(x, z)

		return generated, z

		