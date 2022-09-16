# Cross Aggregation Transformer for Image Restoration

Zheng Chen, Yulun Zhang, Jinjin Gu, Yongbing Zhang, Linghe Kong, and Xin Yuan

---

> **Abstract:** *Recently, Transformer architecture has been introduced into image restoration to replace convolution neural network (CNN) with surprising results. Considering the high computational complexity of Transformer with global attention, some methods use the local square window to limit the scope of self-attention. However, these methods lack direct interaction among different windows, which limits the establishment of long-range dependencies. To address the above issue, we propose a new image restoration model, Cross Aggregation Transformer (CAT). The core of our CAT is the Rectangle-Window Self-Attention (Rwin-SA), which utilizes horizontal and vertical rectangle window attention in different heads parallelly to expand the attention area and aggregate the features cross different windows. We also introduce the Axial-Shift operation for different window interactions. Furthermore, we propose the Locality Complementary Module to complement the self-attention mechanism, which incorporates the inductive bias of CNN (e.g., translation invariance and locality) into Transformer, enabling global-local coupling. Extensive experiments demonstrate that our CAT outperforms recent state-of-the-art methods on several image restoration applications.* 
>
> <p align="center">
> <img width="800" src="figs/git.png">
> </p>

## Dependencies

- See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run Restormer.

## Contents

1. [Datasets and Visual Results](#Datasets and Visual Results)
1. [Training](#Training)
1. [Testing](#Testing)
1. [Results](#Results)
1. [Citation](#Citation)
1. [Acknowledgements](#Acknowledgements)

---

## Datasets and Visual Results


Used training and testing sets can be downloaded as follows:

| Task                                          |                         Training Set                         |                         Testing Set                          |                        Visual Results                        |
| :-------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| image SR                                      | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) | Set5 + Set14 + BSD100 + Urban100 + Manga109 [download all](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) | [here](https://drive.google.com/file/d/1VfeQx0_ThWEtkLsZyPGm4tD0w8SM1tou/view?usp=sharing) |
| grayscale JPEG compression artifact reduction | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images) | grayscale: Classic5 +LIVE + Urban100 [download all](https://drive.google.com/file/d/1cIBdCY99qORfUiMw8sK4lX1KihEGHrYG/view?usp=sharing) | [here](https://drive.google.com/file/d/1YlmySgU1gAdDcuS4fRRHIPpw78NrcMxF/view?usp=sharing) |

## Training

### Image SR

- Download training (DIV2K, Flickr2K) and testing (Set5, Set14, BSD100, Urban100, Manga109) datasets, place them in '/datasets'. 

- Run the folloing scripts. The testing configuration is in 'options/Train'.

  ```shell
  # CAT-A, SR, input=64x64, output=256x256, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_sr_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_sr_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_sr_x4.yml --launcher pytorch
  
  # CAT-R, SR (X4), input=64x64, output=256x256, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_sr_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_sr_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_sr_x4.yml --launcher pytorch
  
  # CAT-R-2, SR (X4), input=64x64, output=256x256, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_2_sr_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_2_sr_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_2_sr_x4.yml --launcher pytorch
  
  # CAT-A-2, SR (X4), input=64x64, output=256x256, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_2_sr_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_2_sr_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_2_sr_x4.yml --launcher pytorch
  ```

- The training experiment is in 'experiments'.

### JPEG Compression Artifact Reduction

- Download training (DIV2K, Flickr2K, WED, BSD) and testing (Classic5, LIVE, Urban100) datasets, place them in '/datasets'.

- Run the folloing scripts. The testing configuration is in 'options/Train'.

  **You can change the training configuration in YML file, like 'train_CAT_A_sr_x4.yml'.**

  ```shell
  # CAT-A, CAR, input=128x128, output=128x128, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_car_q40.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_car_q30.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_car_q20.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_car_q40.yml --launcher pytorch
  ```

- The training experiment is in 'experiments'.

## Testing

### Image SR

- Download the pre-trained [models](https://drive.google.com/drive/folders/1Pd4tuE3f84aY5bcjR8KA5FshAT3-MXLB?usp=sharing) and place them in `experiments/pretrained_models/`.

   We provide some models for image SR: CAT-R, CAT-A, CAT-A, and CAT-R-2 ($\times$2, $\times$3, $\times$4).

- Run the folloing scripts. The testing configuration is in 'options/Test'.

   ```shell
   # CAT-R, SR (X4), produces results in Table 2 of the main paper
   python basicsr/test.py -opt options/Test/test_CAT_R_sr_x2.yml
   python basicsr/test.py -opt options/Test/test_CAT_R_sr_x3.yml
   python basicsr/test.py -opt options/Test/test_CAT_R_sr_x4.yml
   
   # CAT-A, SR (X4), produces results in Table 2 of the main paper
   python basicsr/test.py -opt options/Test/test_CAT_A_sr_x2.yml
   python basicsr/test.py -opt options/Test/test_CAT_A_sr_x3.yml
   python basicsr/test.py -opt options/Test/test_CAT_A_sr_x4.yml
   
   # CAT-R-2, SR (X4), produces results in Table 1 of the supplementary material
   python basicsr/test.py -opt options/Test/test_CAT_R_2_sr_x2.yml
   python basicsr/test.py -opt options/Test/test_CAT_R_2_sr_x3.yml
   python basicsr/test.py -opt options/Test/test_CAT_R_2_sr_x4.yml
   
   # CAT-A-2, SR (X4), produces results in Table 1 of the supplementary material
   python basicsr/test.py -opt options/Test/test_CAT_A_2_sr_x2.yml
   python basicsr/test.py -opt options/Test/test_CAT_A_2_sr_x3.yml
   python basicsr/test.py -opt options/Test/test_CAT_A_2_sr_x4.yml
   ```

- The output is in 'results'.

### JPEG Compression Artifact Reduction

- Download the pre-trained [models](https://drive.google.com/drive/folders/18414_dEErUhZyeHfWGaSa6PLesM3X3ie?usp=sharing) and place them in `experiments/pretrained_models/`.

  We provide some models for JPEG compression artifact reduction: CAT ($q$10, $q$20, $q$30, $q$40).

- Run the folloing scripts. The testing configuration is in 'options/Test'.

  ```shell
  # CAT-A, SR (X4), produces results in Table 3 of the main paper
  python basicsr/test.py -opt options/Test/test_CAT_car_q10.yml
  python basicsr/test.py -opt options/Test/test_CAT_car_q20.yml
  python basicsr/test.py -opt options/Test/test_CAT_car_q30.yml
  python basicsr/test.py -opt options/Test/test_CAT_car_q40.yml
  ```

- The output is in 'results'.

## Results

<details>
<summary>Image SR (click to expan)</summary>
<p align="center">
  <img width="900" src="figs/SR-1.png">
  <img width="900" src="figs/SR-2.png">
</p>
</details>

<details>
<summary>JPEG Compression Artifact Reduction (click to expan)</summary>
<p align="center">
  <img width="900" src="figs/CAR-1.png">
  <img width="900" src="figs/CAR-2.png">
</p>
</details>

## Citation

    

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR).
