import argparse

def options():
	p = argparse.ArgumentParser(description = 'Arguments for image2image-translation training.')
	p.add_argument('--rec-type', help = 'reconstruction loss type (pixel / feature)', default = 'pixel')
	p.add_argument('--gan-loss-type', help = 'gan loss type', default = 'HINGEGAN')

	p.add_argument('--rec-weight', type = float, help = 'rec loss weight', default = 10)
	p.add_argument('--vgg-weight', type = float, help = 'vgg loss weight', default = 10)
	p.add_argument('--kl-weight', type = float, help = 'kl divergence weight', default = 0.05)
	p.add_argument('--ds-weight', type = float, help = 'ds loss weight', default = 8)

	p.add_argument('--decay-start-epoch', type = int, help = 'what epoch will the learning rate decay start', default = -1)
	p.add_argument('--global-freeze-epoch', type = int, help = 'how many epochs will the global network be frozen', default = 10)

	p.add_argument('--epoch', type = int, help = 'epoch num', default = 200)
	p.add_argument('--bs', type = int, help = 'batch size', default = 1)
	p.add_argument('--use-ttur', action = 'store_true', help = 'use TTUR')
	p.add_argument('--lr', type = float, help = 'learning rate', default = 0.0002)
	p.add_argument('--beta1', type = float, help = 'beta1 parameter for the Adam optimizer', default = 0.0)
	p.add_argument('--beta2', type = float, help = 'beta2 parameter for the Adam optimizer', default = 0.9)
	p.add_argument('--ic', type = int, help = 'input channel num (when using in label mode, consider the case of unk)', default = 3)
	p.add_argument('--oc', type = int, help = 'output channel num', default = 3)
	p.add_argument('--height', type = int, help = 'image height (2^n)', default = 256)
	p.add_argument('--width', type = int, help = 'image width (2^n)', default = 256)
	p.add_argument('--network-mode', help = 'model depth (normal/more/most)', default = 'normal')

	p.add_argument('--net-G-type', help = 'Generator Architecture Type', default = 'UNet')
	p.add_argument('--use-sn-G', action = 'store_true', help = 'use Spectral Normalization in the Generator')
	p.add_argument('--norm-type-G', help = 'normalization type in the Generator', default = 'instancenorm')
	p.add_argument('--z-dim', type = int, help = 'latent vector size', default = 0)

	p.add_argument('--net-D-type', help = 'Discriminator Architecture Type', default = 'PatchGan')
	p.add_argument('--use-sn-D', action = 'store_true', help = 'use Spectral Normalization in the Discriminator')
	p.add_argument('--norm-type-D', help = 'normalization type in the Discriminator', default = 'instancenorm')
	p.add_argument('--use-sigmoid', action = 'store_true', help = 'use Sigmoid in the last layer of Discriminator')
	p.add_argument('--D-scale-num', type = int, help = 'number of multiscale Discriminator', default = 1)

	p.add_argument('--use-encoder', action = 'store_true', help = 'use encoder')

	p.add_argument('--weight-init-type', help = 'network weight init type (xavier, normal)', default = 'xavier')

	p.add_argument('--print-freq', type = int, help = 'prints the loss value every few iterations', default = 100)
	p.add_argument('--vis-freq', type = int, help = 'saves the visualization every few iterations', default = 100)
	p.add_argument('--vis-pth', help = 'path to save the visualizations', default = 'visualizations/')
	p.add_argument('--model-pth', help = 'path to save the final model', default = 'models/model.pth')

	p.add_argument('--data-type', help = 'dataloader type (image / label)', default = 'image')
	p.add_argument('--trn-src-pth', help = 'train src dataset path', default = 'data/train/src')
	p.add_argument('--trn-trg-pth', help = 'train trg dataset path', default = 'data/train/trg')
	p.add_argument('--val-src-pth', help = 'val src dataset path', default = 'data/val/src')
	p.add_argument('--val-trg-pth', help = 'val trg dataset path', default = 'data/val/trg')
	p.add_argument('--label-pth', help = 'label path', default = 'data/label.pkl')

	p.add_argument('--num-workers', type = int, help = 'num workers for the dataloader', default = 10)
	p.add_argument('--grad-acc', type = int, help = 'split the batch into n steps', default = 1)
	p.add_argument('--multigpu', action = 'store_true', help = 'use multiple gpus')

	args = p.parse_args()
	return args