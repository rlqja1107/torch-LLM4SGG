## Dataset Instruction  

### Directory Structure  

```
root  
├── dataset 
│   ├── COCO     
│   │    │── captions_train2017.json    
│   │    │── captions_val2017.json
│   │    │── COCO_triplet_labels.npy    
│   │    └── images 
|   │         └── *.png        
│   ├── VG
│   │    │── image_data.json
│   │    │── VG-SGG-with-attri.h5
│   │    │── VG-SGG-dicts-with-attri.json
│   │    └── images
│   │         └── *.png 
│   ├── GQA
│   │    │── GQA_200_ID_Info.json
│   │    │── GQA_200_Train.json
│   │    │── GQA_200_Test.json
│   │    └── images
│   │         └── *.png
│   ├── CC
│   │    │── Train_GCC-training.tsv
│   │    │── cc_triplet_labels.npy
│   │    │── cc_meta_information.json
│   │    └── images
│   │         └── *.png  
│   ├── VG_Caption
│   │    └── region_descriptions.json
```
    

### Training data (Caption dataset)

We use training datasets for `COCO`, `CC`, and `Visual Genome caption` datasets. 
You can conveniently download each set of training data using shell code. For detailed information, please refer to each following links.

``` python  
# DATASET: COCO, CC, VG_Caption
bash dataset/{DATASET}/download.sh dataset/{DATASET}
```

#### Detailed Download instructions
* <img src="../figure/coco-logo2.png" width="15"> COCO: [README.md](COCO/README.md) 
* CC: [README.md](CC/README.md)
* <img src="../figure/vg-logo.png" width="14"> Visual Genome Caption: [README.md](VG_Caption/README.md)


### Test data 
For evaluation, we use `Visual Genome (VG)` and `GQA` datasets. 

<!--※ *Unfortunately, raw images in GQA dataset are not available since the corresponding homepage is expired.*-->

``` python  
# DATASET: VG, GQA
bash dataset/{DATASET}/download.sh dataset/{DATASET}
```

#### Detailed Download instructions
* <img src="../figure/vg-logo.png" width="14"> Visual Genome: [README.md](VG/README.md)  
* <img src="../figure/gqa-logo.png" width="15"> GQA: [README.md](GQA/README.md) 