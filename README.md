# Global HRTF personalization using Anthropometric Measurements (AES 2021)

[Paper link][2]

Authors: Yuxiang Wang, You Zhang, Zhiyao Duan, Mark Bocko

## Basic information
Dataset: HUTUBS [[download here][1]]

Programming: MATLAB, Python (Pytorch)

## Preprocessing with SHT
Please first download the HUTUBS dataset and unzip all sub-folders into this repo.
We will use `HRIR` and `Antrhopometric measures` sub-folders.

## Deep learning model
### Dependencies
numpy

pandas

scipy

tqdm

pytorch


### Train the model
Please take care of the arguments in `train.py`, especially specify the `anthro_mat_path` where you put the `AntrhopometricMeasures.csv` of the HUTUBS dataset, and the `out_fold` where you want the trained model to be stored at.

### Test the pretrained model

## Citation
```
@article{wang2021global,
author={wang, yuxiang and zhang, you and duan, zhiyao and bocko, mark},
journal={journal of the audio engineering society},
title={global hrtf personalization using anthropometric measures},
year={2021},
volume={},
number={},
pages={},
doi={},
month={may},}
```



[1]: https://depositonce.tu-berlin.de/handle/11303/9429
[2]: https://www.aes.org/e-lib/browse.cfm?elib=21095
