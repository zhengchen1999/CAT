import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from cat_unet_arch import CAT_Unet
import scipy.io as sio

parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('--input_dir', default='../datasets/SIDD/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='../results/Real_Denoising/SIDD/', type=str, help='Directory for results')
parser.add_argument('--weights', default='../experiments/pretrained_models/CAT.pth', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'RealDenoising_CAT.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

model_restoration = CAT_Unet(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Process data
filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /=255.
restored = np.zeros_like(Inoisy)
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
            # print(noisy_patch.size())
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i,k,:,:,:] = restored_patch

# save denoised data
sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored,})
