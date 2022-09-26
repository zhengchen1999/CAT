For training and testing, recollect all datasets as the following form:
```shell
|-- datasets
    # image SR - train
    |-- DF2K
        |-- HR
        |-- LR_bicubic
            |-- X2
            |-- X3
            |-- X4
    # image SR - test
    |-- benchmark
        |-- Set5
            |-- HR
            |-- LR_bicubic
                |-- X2
                |-- X3
                |-- X4
        |-- Set14
            |-- HR
            |-- LR_bicubic
                |-- X2
                |-- X3
                |-- X4
        |-- B100
            |-- HR
            |-- LR_bicubic
                |-- X2
                |-- X3
                |-- X4
        |-- Urban100
            |-- HR
            |-- LR_bicubic
                |-- X2
                |-- X3
                |-- X4
        |-- Manga109
            |-- HR
            |-- LR_bicubic
                |-- X2
                |-- X3
                |-- X4     
                
    # grayscale JPEG compression artifact reduction - train & test
    |-- CAR
        |-- DFWB_HQ
        |-- DFWB_LQ
            |-- 10
            |-- 20
            |-- 30
            |-- 40
        |-- classic5
            |-- Classic5_HQ
            |-- Classic5_LQ
                |-- 10
                |-- 20
                |-- 30
                |-- 40
        |-- LIVE1
            |-- LIVE1_HQ
            |-- LIVE1_LQ
                |-- 10
                |-- 20
                |-- 30
                |-- 40
        |-- Urban100
            |-- Urban100_HQ
            |-- Urban100_LQ
                |-- 10
                |-- 20
                |-- 30
                |-- 40
    # real image denoising - test
    |-- test
        |-- SIDD
            |-- ValidationGtBlocksSrgb.mat
            |-- ValidationNoisyBlocksSrgb.mat
        |-- DND
            |-- info.mat
            |-- ValidationNoisyBlocksSrgb
                |-- 0001.mat
                |-- 0002.mat
                ：  
                |-- 0050.mat
|-- Restormer
    # real image denoising - train & val
    |-- datasets
        |-- Dowloads
            |-- train
                |-- 0001_001_S6_00100_00060_3200_L
                ：  
                |-- 0200_010_GP_01600_03200_5500_N
        |-- train
            |-- SIDD
                |-- target_crops
                |-- input_crops
        |-- val
            |-- SIDD
                |-- target_crops
                |-- input_crops
```
