
from argparse import Namespace

import torch
import numpy as np
import random

from src.train import train
from src.utils import get_simu_data

import wandb

wb = True

opts = Namespace(
	batch_size = 256,
	mode = 'train',
	lr = 5e-2,
	epochs = 200,
	grad_clip = False,
	mxAlpha = 4,
	mxBeta = 0.5,
	mxTemp = 1,
	lmbda = 1e-1,
	MMD_sigma = 1000,
	kernel_num = 10,
	matched_IO = False,
	latdim = 4,
	seed = 12
)

#device = 'cuda:0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', device)

torch.manual_seed(opts.seed)
np.random.seed(opts.seed)
random.seed(opts.seed)

# Train and Test Data
dataloader, dataloader2, dim, cdim, ptb_targets, nonlinear = get_simu_data(batch_size=opts.batch_size, mode=opts.mode)

if wb:
	wandb.init(
					project = 'discrepancy_vae',
					entity = 'mariamartinezga',
					name = 'simulation_images',
					config=opts
				)

# Dimensionality of the data
opts.dim = dim
if opts.latdim is None:
	opts.latdim = cdim
opts.cdim = cdim

train(dataloader, opts, device, './result/', log=True, simu=True, nonlinear=nonlinear, order=[0,1,2,3])

# Evalution
savedir = './result/' 

model = torch.load(f'{savedir}/best_model.pt', weights_only=False)

model.eval()
order = []
c_set = np.eye(4)
for i in range(4):
    c = c_set[i,:]
    c = torch.from_numpy(c).to(device).double().unsqueeze(0)

    bc,csz = model.c_encode(c, temp=1)
    order.append(bc.argmax().item())

print(order)

print(torch.triu(model.G, diagonal=1))