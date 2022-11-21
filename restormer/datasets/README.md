For real image denoising training, recollect datasets as the following form:

```shell
# real image denoising - train & val
|-- datasets
    |-- SIDD
        |-- train
            |-- target_crops
            |-- input_crops  
        |-- val
            |-- target_crops
            |-- input_crops   
        # the raw data of the SIDD-train
        |-- raw
            |-- 0001_001_S6_00100_00060_3200_L
            ：  
            |-- 0200_010_GP_01600_03200_5500_N
```

You can download [the complete training datasets](https://drive.google.com/file/d/1dbdRgMljekABcXs_tuKq4ilSO2GRcFvo/view?usp=share_link) we have collected. 

The [raw](https://drive.google.com/file/d/1FT1gzmNkT4NklPM1Hv_YP_S2zltCOz94/view?usp=share_link) contains the raw data of the SIDD-train. You can re-generate the training dataset use [generate_patches_sidd.py](../generate_patches_sidd.py).

The val dataset can be downloaded [here](https://drive.google.com/file/d/1S5Oz2HE7R5CbXvfs7PMlmBdAYPnByOT7/view?usp=share_link).

