For real image denoising training, recollect datasets as the following form:

```shell
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

