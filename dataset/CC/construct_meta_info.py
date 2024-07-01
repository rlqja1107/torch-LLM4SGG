import os
import json
from PIL import Image


data = json.load(open("dataset/CC/aligned_triplet_info_cc4vg_grounded.json",'r'))

path = 'dataset/CC/training'
file_name = os.listdir(path)
img_id_list = list(data['text_scene_graph'].keys())

cc_img_info = {}
for k in img_id_list:
    cc_img_info[k] = {}
    pil_image = Image.open(path+"/"+f"{k}.jpg").convert("RGB")
    width = pil_image.width
    height = pil_image.height
    file_name = f"{k}.jpg"
    img_id = int(k)
    cc_img_info[k]['file_name'] = file_name
    cc_img_info[k]['width'] = width
    cc_img_info[k]['height'] = height
    cc_img_info[k]['id'] = img_id
    
with open("dataset/CC/cc_meta_information.json", 'w') as f:
    json.dump(cc_img_info, f)