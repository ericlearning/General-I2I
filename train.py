import os
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from options.options import options
from dataloaders.I2I_dataloader import Dataset
from trainer.trainer import Image2Image_Trainer
from utils.network_utils import *
from utils.visualization_utils import *

opt = options()
ds = Dataset(opt)
trn_dl, val_dl = ds.get_loader(opt.bs)
iter_num, total_iter_num = len(trn_dl), len(trn_dl) * opt.epoch
data_type, label_path = opt.data_type, opt.label_pth
trainer = Image2Image_Trainer(opt, iter_num)

vis_z = generate_noise(3, opt.z_dim)
save_cnt = 0
for epoch in range(opt.epoch):
	stage = get_stage(epoch, opt.epoch, trainer.network.netG_type, opt.global_freeze_epoch)
	for i, data in enumerate(tqdm(trn_dl)):
		errD, errG = trainer.step(data, stage)

		if(i % opt.print_freq == 0):
			print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
				  %(epoch+1, opt.epoch, i+1, len(trn_dl), float(errD), float(errG)))

		if(i % opt.vis_freq == 0):
			sample_images_list = get_sample_images_list(trainer, val_dl, vis_z, stage, label_path, data_type)
			if(opt.z_dim > 0): plot_img = get_display_samples(sample_images_list, 9, 3)
			else: plot_img = get_display_samples(sample_images_list, 3, 3)

			img_fn = str(save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg'
			img_pth = os.path.join(opt.vis_pth, img_fn)
			save_cnt += 1
			cv2.imwrite(img_pth, plot_img)

trainer.save(opt.model_pth)