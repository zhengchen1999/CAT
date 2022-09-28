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
        |-- download
            |-- 0001_001_S6_00100_00060_3200_L
            ：  
            |-- 0200_010_GP_01600_03200_5500_N
```

You can download the complete datasets we have collected. 

The `download/` contains the raw data of the SIDD-train. You can re-generate the training dataset use [generate_patches_sidd.py](../generate_patches_sidd.py).
