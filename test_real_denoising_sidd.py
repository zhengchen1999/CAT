# The implementation builds on Restormer code https://github.com/swz30/Restormer
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from basicsr.archs.cat_unet_arch import CAT_Unet
import scipy.io as sio

import cv2
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='Real Image Denoising using CAT')

parser.add_argument('--input_dir', default='datasets/real-DN/SIDD/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='results/test_CAT_Real_DN/SIDD/', type=str, help='Directory for results')
parser.add_argument('--weights', default='experiments/pretrained_models/Real-DN/Real_DN_CAT.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'options/test/test_CAT_RealDenoising.yml'
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

if args.save_images:
    result_dir_png = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_png, exist_ok=True)

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

            if args.save_images:
                save_file = os.path.join(result_dir_png, '%04d_%02d.png'%(i+1,k+1))
                cv2.imwrite(save_file, cv2.cvtColor(img_as_ubyte(restored_patch), cv2.COLOR_RGB2BGR))

# save denoised data
sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored,})
