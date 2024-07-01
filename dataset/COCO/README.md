## COCO Caption Dataset

For default setting, we use image caption with its image in `COCO` dataset. Please download the [COCO](https://cocodataset.org/#download) dataset and put the corresponding files into *dataset/COCO* directory. The name of files in url are:

> 2017 Traing images [118K/18GB]  
> 2017 Val images [5K/1GB]  
> 2017 Train/Val annotations [241MB]   

Note that after downloading the raw images, please combine them into *dataset/COCO/images* directory. For a fair comparison, we use 64K images, following the previous studies ([SGNLS](https://github.com/YiwuZhong/SGG_from_NLS), [Li et al, MM'22](https://github.com/xcppy/WS-SGG)). Please download a [file](https://drive.google.com/file/d/1kXVsecabQig2aC8KUpDb831J5_e-aJrl/view?usp=sharing) including the image id of 64K images. 


### Shell code
Alternatively, you can run shell code to download all necessary files.
``` python  
bash dataset/COCO/download.sh dataset/COCO
```