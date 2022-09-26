# Cross Aggregation Transformer for Image Restoration

Zheng Chen, Yulun Zhang, Jinjin Gu, Yongbing Zhang, Linghe Kong, and Xin Yuan, "Cross Aggregation Transformer for Image Restoration", NeurIPS, 2022

---

> **Abstract:** *Recently, Transformer architecture has been introduced into image restoration to replace convolution neural network (CNN) with surprising results. Considering the high computational complexity of Transformer with global attention, some methods use the local square window to limit the scope of self-attention. However, these methods lack direct interaction among different windows, which limits the establishment of long-range dependencies. To address the above issue, we propose a new image restoration model, Cross Aggregation Transformer (CAT). The core of our CAT is the Rectangle-Window Self-Attention (Rwin-SA), which utilizes horizontal and vertical rectangle window attention in different heads parallelly to expand the attention area and aggregate the features cross different windows. We also introduce the Axial-Shift operation for different window interactions. Furthermore, we propose the Locality Complementary Module to complement the self-attention mechanism, which incorporates the inductive bias of CNN (e.g., translation invariance and locality) into Transformer, enabling global-local coupling. Extensive experiments demonstrate that our CAT outperforms recent state-of-the-art methods on several image restoration applications.* 
>
> <p align="center">
> <img width="800" src="figs/git.png">
> </p>

## Dependencies

- See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run Restormer.

## TODO

* [x] Image SR
* [x] JPEG compression artifact reduction
* [x] Image Denoising
* [ ] Image Deblurring
* [ ] Image Deraining

## Contents

1. [Datasets](#Datasets)
1. [Models](#Models)
1. [Training](#Training)
1. [Testing](#Testing)
1. [Results](#Results)
1. [Citation](#Citation)
1. [Acknowledgements](#Acknowledgements)

---

## Datasets


Used training and testing sets can be downloaded as follows:

| Task                                          |                         Training Set                         |                         Testing Set                          |                        Visual Results                        |
| :-------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| image SR                                      | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) | Set5 + Set14 + BSD100 + Urban100 + Manga109 [download all](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) | [here](https://drive.google.com/drive/folders/122LBzNSuc-YwzyTzA2VL9mXSJPWBxICZ?usp=sharing) |
| grayscale JPEG compression artifact reduction | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images) | grayscale: Classic5 +LIVE + Urban100 [download all](https://drive.google.com/file/d/17hwSblurN93ndKFRFQQdoRgB-6pFmGtd/view?usp=sharing) | [here](https://drive.google.com/drive/folders/1xwBMPRUIAnpjAynEr9GI8Wenl6D-J8i3?usp=sharing) |
| real image denoising                          | [SIDD](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing) (320 training images) | [SIDD](https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/view?usp=sharing) + [DND](https://drive.google.com/file/d/1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G/view?usp=sharing) | [here](https://drive.google.com/drive/folders/14chIIFh6uG4M-aOyJcu6mYjDIpm4zE5t?usp=sharing) |

## Models

|  Task   | Method  | Params (M) | FLOPs (G) | Dataset  | PSNR  |  SSIM  |                          Model Zoo                           |                        Visual Results                        |
| :-----: | :------ | :--------: | :-------: | :------: | :---: | :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   SR    | CAT-R   |   16.60    |   292.7   | Urban100 | 27.45 | 0.8254 | [Google Drive](https://drive.google.com/drive/folders/1oBCa_ZmKQnqtkgfk2b5P5nGi-c2oH6Ez?usp=sharing) | [Google Drive](https://drive.google.com/file/d/108IvgnaibEGtPIcefovh_drGGbuWNOqv/view?usp=sharing) |
|   SR    | CAT-A   |   16.60    |   360.7   | Urban100 | 27.89 | 0.8339 | [Google Drive](https://drive.google.com/drive/folders/1Xm4xQXI74gZcPwgmQHw1qbgdEA0kVCSP?usp=sharing) | [Google Drive](https://drive.google.com/file/d/14Jy3y7ILGb1W-HqHvobQCeDT4JEfBr0w/view?usp=sharing) |
|   SR    | CAT-R-2 |   11.93    |   216.3   | Urban100 | 27.59 | 0.8285 | [Google Drive](https://drive.google.com/drive/folders/175wdTqjpURS7TSRIj3DODVI_ppktqdJu?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1pRoALQRTfngmSURcIWGemHp0ozpJU7rk/view?usp=sharing) |
|   SR    | CAT-A-2 |   16.60    |   387.9   | Urban100 | 27.99 | 0.8357 | [Google Drive](https://drive.google.com/drive/folders/1oBCa_ZmKQnqtkgfk2b5P5nGi-c2oH6Ez?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1L-Qw3Jbd4yghuJIGB9n0PMGO0VpSwD2v/view?usp=sharing) |
|   CAR   | CAT     |   16.20    |   346.4   |  LIVE1   | 29.89 | 0.8295 | [Google Drive](https://drive.google.com/drive/folders/18414_dEErUhZyeHfWGaSa6PLesM3X3ie?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/1xwBMPRUIAnpjAynEr9GI8Wenl6D-J8i3?usp=sharing) |
| real-DN | CAT     |   25.77    |   53.2    |   SIDD   | 40.01 | 0.9600 | [Google Drive](https://drive.google.com/drive/folders/1rkNeKeGiZqKit0M_AxFx1yfGu6z1ahgc?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/14chIIFh6uG4M-aOyJcu6mYjDIpm4zE5t?usp=sharing) |

The performance are reported on Urban100 (x4, SR), LIVE1 (q=10, CAR), and SIDD (real-DN). The test size of FLOPS is 128 x 128.

## Training

### Image SR

- Download training ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)) and testing ([Set5, Set14, BSD100, Urban100, Manga109](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing)) datasets, place them in '/datasets'.

- Run the folloing scripts. The testing configuration is in 'options/train'.

  ```shell
  # CAT-R, SR, input=64x64, output=256x256, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_sr_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_sr_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_sr_x4.yml --launcher pytorch
  
  # CAT-A, SR, input=64x64, output=256x256, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_sr_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_sr_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_sr_x4.yml --launcher pytorch
  
  # CAT-R-2, SR, input=64x64, output=256x256, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_2_sr_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_2_sr_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_2_sr_x4.yml --launcher pytorch
  
  # CAT-A-2, SR, input=64x64, output=256x256, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_2_sr_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_2_sr_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_2_sr_x4.yml --launcher pytorch
  ```

- The training experiment is in 'experiments'.

### JPEG Compression Artifact Reduction

- Download training ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz), [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)) and testing ([Classic5, LIVE, Urban100](https://drive.google.com/file/d/17hwSblurN93ndKFRFQQdoRgB-6pFmGtd/view?usp=sharing)) datasets, place them in '/datasets'.

- Run the folloing scripts. The testing configuration is in 'options/train'.

  ```shell
  # CAT, CAR, input=128x128, output=128x128, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_car_q40.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_car_q30.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_car_q20.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_car_q40.yml --launcher pytorch
  ```
  
- The training experiment is in 'experiments'.

### Real Image Denoising

- Cd to 'Restormer'. For real image denoising, we train CAT directly with [Restormer](https://github.com/swz30/Restormer) as the codebase.

- Download training ([SIDD train](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing)) datasets and val ([SIDD val](https://drive.google.com/file/d/1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ/view?usp=sharing)) datasets. Unzip and place SIDD train in '/datasets/Dowloads' ('Restormer/datasets/Dowloads'). Run the folloing scripts to generate image patches. Place SIDD val in '/datasets'. ('Restormer/datasets')

  ```shell
  python generate_patches_sidd.py 
  ```

- Run the folloing scripts. The testing configuration is in 'options' ('Restormer/options').

  ```shell
  # CAT, CAR, Progressive Learning, 8 GPUs
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train_RealDenoising_CAT.yml --launcher pytorch
  ```

- The training experiment is in 'experiments' ('Restormer/experiments').

## Testing

### Image SR

- Download the pre-trained [models](https://drive.google.com/drive/folders/1Pd4tuE3f84aY5bcjR8KA5FshAT3-MXLB?usp=sharing) and place them in `experiments/pretrained_models/`.

   We provide some models for image SR: CAT-R, CAT-A, CAT-A, and CAT-R-2 (x2, x3, x4).

- Download testing ([Set5, Set14, BSD100, Urban100, Manga109](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing)) datasets, place them in '/datasets'.

- Run the folloing scripts. The testing configuration is in 'options/test'.

   ```shell
   # CAT-R, SR, produces results in Table 2 of the main paper
   python basicsr/test.py -opt options/Test/test_CAT_R_sr_x2.yml
   python basicsr/test.py -opt options/Test/test_CAT_R_sr_x3.yml
   python basicsr/test.py -opt options/Test/test_CAT_R_sr_x4.yml
   
   # CAT-A, SR, produces results in Table 2 of the main paper
   python basicsr/test.py -opt options/Test/test_CAT_A_sr_x2.yml
   python basicsr/test.py -opt options/Test/test_CAT_A_sr_x3.yml
   python basicsr/test.py -opt options/Test/test_CAT_A_sr_x4.yml
   
   # CAT-R-2, SR, produces results in Table 1 of the supplementary material
   python basicsr/test.py -opt options/Test/test_CAT_R_2_sr_x2.yml
   python basicsr/test.py -opt options/Test/test_CAT_R_2_sr_x3.yml
   python basicsr/test.py -opt options/Test/test_CAT_R_2_sr_x4.yml
   
   # CAT-A-2, SR, produces results in Table 1 of the supplementary material
   python basicsr/test.py -opt options/Test/test_CAT_A_2_sr_x2.yml
   python basicsr/test.py -opt options/Test/test_CAT_A_2_sr_x3.yml
   python basicsr/test.py -opt options/Test/test_CAT_A_2_sr_x4.yml
   ```

- The output is in 'results'.

### JPEG Compression Artifact Reduction

- Download the pre-trained [models](https://drive.google.com/drive/folders/18414_dEErUhZyeHfWGaSa6PLesM3X3ie?usp=sharing) and place them in `experiments/pretrained_models/`.

  We provide some models for JPEG compression artifact reduction: CAT (q10, q20, q30, q40).

- Download testing ([Classic5, LIVE, Urban100](https://drive.google.com/file/d/17hwSblurN93ndKFRFQQdoRgB-6pFmGtd/view?usp=sharing)) datasets, place them in '/datasets'.

- Run the folloing scripts. The testing configuration is in 'options/test'.

  ```shell
  # CAT-A, SR, produces results in Table 3 of the main paper
  python basicsr/test.py -opt options/Test/test_CAT_car_q10.yml
  python basicsr/test.py -opt options/Test/test_CAT_car_q20.yml
  python basicsr/test.py -opt options/Test/test_CAT_car_q30.yml
  python basicsr/test.py -opt options/Test/test_CAT_car_q40.yml
  ```

- The output is in 'results'.

### Real Image Denoising

- Download the pre-trained [models](https://drive.google.com/drive/folders/1rkNeKeGiZqKit0M_AxFx1yfGu6z1ahgc?usp=sharing) and place them in `experiments/pretrained_models/`.

- Download testing ([SIDD](https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/view?usp=sharing), [DND](https://drive.google.com/file/d/1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G/view?usp=sharing)) datasets, place them in '/datasets'.

- Run the folloing scripts.

  ```shell
  # CAT, real-DN, produces results in Table 4 of the main paper
  # testing on SIDD
  python test_real_denoising_sidd.py --save_images
  evaluate_sidd.m
  
  # testing on DND
  python test_real_denoising_dnd.py --save_images
  ```

- The output is in 'results'.

## Results

<details>
<summary>Image SR (click to expan)</summary>

- results in Table 2 of the main paper

<p align="center">
  <img width="900" src="figs/SR-1.png">
</p>

- results in Table 1 of the supplementary material

<p align="center">
  <img width="900" src="figs/SR-2.png">
</p>

</details>

<details>
<summary>JPEG Compression Artifact Reduction (click to expan)</summary>

- results in Table 3 of the main paper

<p align="center">
  <img width="900" src="figs/CAR-1.png">
  <img width="900" src="figs/CAR-2.png">
</p>
</details>

<details>
<summary>Real Image Denoising (click to expan)</summary>

- results in Table 4 of the main paper

<p align="center">
  <img width="900" src="figs/Real-DN.png">
</p>
</details>

## Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).
```
@inproceedings{chen2022cross,
    title={Cross Aggregation Transformer for Image Restoration},
    author={Chen, Zheng and Zhang, Yulun and Gu, Jinjin and Zhang, Yongbing and Kong, Linghe and Yuan, Xin},
    booktitle={NeurIPS},
    year={2022}
}
```

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR) and [Restormer](https://github.com/swz30/Restormer).
