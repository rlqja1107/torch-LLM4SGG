
# Triplet Extraction Process

## <img src="../figure/extraction.png" width="20"> **Triplet Extraction Process via LLM - VG**  

To utilize *gpt-3.5-turbo* in ChatGPT, please insert your openai key which is obtained from [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) 

Please follow step by step to obtain localized triplets.


### (1/4) Chain-1: Triplet Extraction via LLM

Since triplet extraction via LLM is based on openAI's API, the code can be runned in parallel. For example, 10,000 images can be divided into 1,000 images with 10 codes. To this end, please change *start* and *end* variables in `.py` code, and name of saved files to avoid overwriting files.

* Extract triplets from original captions  
``` python  
python triplet_extraction_process/extract_triplet_with_original_caption.py {API_KEY}
```   


* Extract triplets from paraphrased captions
``` python  
python triplet_extraction_process/extract_triplet_with_paraphrased_caption.py {API_KEY}
```   
After **Chain-1**, the output files are located in *dataset/COCO* directory. The files containing misaligned triplets can be downloaded as:

* [dataset/COCO/misaligned_triplets_original.pkl](https://drive.google.com/file/d/1eEZ5CqYPJCI_jYaEBCBgESnI4EX4uSgM/view?usp=sharing) 

* [dataset/COCO/misaligned_triplets_paraphrased.pkl](https://drive.google.com/file/d/17cef1jJIw0zpdlm2zlIGo9Ozfp3OnkP7/view?usp=sharing)  

### (2/4) Chain-2: Alignment of Classes in Triplets via LLM

``` python  
python triplet_extraction_process/alignment_classes_vg.py {API_KEY}
```   

After **Chain-2**, the output files are located in *triplet_extraction_process/alignment_dict* directory. The files containing aligned entity/predicate information can be downloaded as:

* [triplet_extraction_process/alignment_dict/aligned_entity_dict_vg.pkl](https://drive.google.com/file/d/1nbmggHoWEywC1GCN4EIPqaIx3NrGJew8/view?usp=sharing)  

* [triplet_extraction_process/alignment_dict/aligned_predicate_dict_vg.pkl](https://drive.google.com/file/d/1ZLtfhPuw4tv56eqH2b6F7SUNgtXOph74/view?usp=sharing)  


### (3/4) Construction of aligned triplets in VS3 format  


``` python  
python triplet_extraction_process/final_preprocess_triplets_vg.py
```   
After **Final** instruction, the output file is located in *dataset/VG* directory. The file containing aligned triplets in VS3 format can be downloaded as follows:

* [dataset/VG/aligned_triplet_info_vg.json](https://drive.google.com/file/d/14UrH6SCH-64CP0vuaoUZaYYN2sXploWi/view?usp=sharing)  

### (4/4) Grounding Unlocalized Triplets  

We follow same code in [VS3](https://github.com/zyong812/VS3_CVPR23/blob/main/tools/data_preprocess/parse_SG_from_COCO_captionV2.py) to ground unlocalized triplets. A pre-trained [GLIP](https://github.com/microsoft/GLIP) model is necessary to ground them. Please put the pre-trained GLIP model to *MODEL* directory. 
``` python  
# Download pre-trained GLIP models
mkdir MODEL
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth -O swin_large_patch4_window12_384_22k.pth
mv glip_tiny_model_o365_goldg_cc_sbu.pth MODEL/
mv swin_large_patch4_window12_384_22k.pth MODEL/
```  

``` python  
# Grounding unlocalized triplets
python tools/data_preprocess/parse_SG_from_COCO_caption_LLM_VG.py
```  

After grounding unlocalized triplets, the output file named *aligned_triplet_info_vg_grounded.json* is located in *dataset/VG* directory. The file of localized triplets can be downloaded as follows:
* [dataset/VG/aligned_triplet_coco4vg_grounded.json](https://huggingface.co/datasets/kb-kim/LLM4SGG/resolve/main/aligned_triplet_coco4vg_grounded.json)  


## <img src="../figure/extraction.png" width="20"> **Triplet Extraction Process via LLM - GQA**

Based on the extracted triplets in [Chain-1](#chain-1-triplet-extraction-via-llm), please run the below codes, similar to the process in [Triplet Extraction Process via LLM - VG](#triplet-extraction-process-via-llm---vg)

``` python  
# Chain-2: Alignment of Classes in Triplets via LLM 
python triplet_extraction_process/alignment_classes_gqa.py {API_KEY}
# Construction of aligned tripelts in VS3 format
python triplet_extraction_process/final_preprocess_triplets_gqa.py
# Grounding Unlocalized Triplets
python tools/data_preprocess/parse_SG_from_COCO_caption_LLM_GQA.py
```   

We provide files regarding GQA dataset.  

* [aligned_entity_dict_gqa.pkl](https://drive.google.com/file/d/1pV5VlcFGmL0PFU7bRqVvNKXhYtiMP9_p/view?usp=sharing), [aligned_predicate_dict_gqa.pkl](https://drive.google.com/file/d/1K-50K2mioFwF1T6VqU-C1sSjNYyXcUza/view?usp=sharing)
* [aligned_triplet_coco4gqa_grounded.json](https://huggingface.co/datasets/kb-kim/LLM4SGG/resolve/main/aligned_triplet_coco4gqa_grounded.json)
