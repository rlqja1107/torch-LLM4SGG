## Conceptual Caption Dataset  

### **1. Download image captions along with URLs of the images**    

On [CC homepage](https://ai.google.com/research/ConceptualCaptions/download), you can download image caption dataset ([Training split](https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250)), whose file name is `Train_GCC-training.tsv`.

### **2. Download raw images**   

We utilize same images used in [SGNLS](https://github.com/YiwuZhong/SGG_from_NLS). With [image index file](https://drive.google.com/file/d/1wxxf5o3hfFKLqjAfvNlP4APWfDShE2e_/view?usp=sharing) provided by [SGNLS](https://github.com/YiwuZhong/SGG_from_NLS), please download raw images.  

``` python
python dataset/CC/download_img.py dataset/CC
```

### **3. Download meta information of CC datasets**  

We construct meta information, which can be downloaded in [link](https://drive.google.com/file/d/1WXGF76JfG7BTAhasqtt6RN_dLmmkgYCm/view?usp=sharing).  

Please refer to `dataset/CC/construct_meta_info.py`  


