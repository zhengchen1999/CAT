# Cross Aggregation Transformer for Image Restoration

This repository is for CAT introduced in the paper.



## Dependencies

- python 3.8
- pyTorch >= 1.8.0
- basicsr (pip install basicsr) For more information, please refer to [BasicSR](https://github.com/XPixelGroup/BasicSR.git).
- timm
- einops

## Test

1. Dwonload models for our paper and place them in 'experiments/pretrained_models'. We provide some models for image SR (x4): CAT-A, CAT-R, and CAT-R-2.

2. Cd to 'CAT', run the folloing scripts. The testing configuration is in 'options/Test'. More detail about YML, please refer to [Training and Testing](https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md).

   **You can change the testing configuration in YML file, like 'test_CAT_A_sr_x4.yml'.**

   ```shell
   # CAT-A, SR (X4), produces results in Table 2 of the main paper
   python basicsr/test.py -opt options/Test/test_CAT_A_sr_x4.yml
   
   # CAT-R, SR (X4), produces results in Table 2 of the main paper
   python basicsr/test.py -opt options/Test/test_CAT_R_sr_x4.yml
   
   # CAT-R-2, SR (X4), produces results in Table 1 of the supplementary material
   python basicsr/test.py -opt options/Test/test_CAT_R_2_sr_x4.yml
   ```

3. The output is in 'results'.

## Train

1. Download DIV2K and Flickr2K training data from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar).

1. Package DIV2K and Flickr2K as DF2K (HR and LR_bicubic), and place them in '/datasets'.

   **You can change the training configuration in YML file, like 'train_CAT_A_sr_x4.yml'.**

   ```shell
   # CAT-A, SR (X4), input=64x64, output=256x256
   PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_A_sr_x4.yml --launcher pytorch
   
   # CAT-R, SR (X4), input=64x64, output=256x256
   PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_sr_x4.yml --launcher pytorch
   
   # CAT-R-2, SR (X4), input=64x64, output=256x256
   PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_CAT_R_2_sr_x4.yml --launcher pytorch
   ```

4. The training experiment is in 'experiments'.

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR.git).

