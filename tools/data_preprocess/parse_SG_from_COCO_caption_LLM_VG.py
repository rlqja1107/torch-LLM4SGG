import os
import json
import torch
import random
import argparse
import requests
import sng_parser
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from maskrcnn_benchmark.data.datasets import VG150Dataset
from maskrcnn_benchmark.data.preprocess_utils import map_caption_concepts_to_vg150_categories

torch.set_num_threads(4)

parser = argparse.ArgumentParser(description="Grounding")
parser.add_argument("--sg_parser", type=str, default='python')
parser.add_argument("--gpu_size", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

def load(url):
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.show()


config_file = "configs/pretrain/glip_Swin_L.yaml"
weight_file = "MODEL/swin_large_patch4_window12_384_22k.pth"

# update the config options with the config file
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.LANGUAGE_BACKBONE.TOKENIZER_LOCAL_FILES_ONLY", True])
cfg.merge_from_list(["MODEL.DYHEAD.NUM_CLASSES", 200])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])


vg_test_dataset = VG150Dataset(split='test',
                         img_dir="dataset/VG/VG_100K",
                         roidb_file="dataset/VG/VG-SGG-with-attri.h5",
                         dict_file="dataset/VG/VG-SGG-dicts-with-attri.json",
                         image_file="dataset/VG/image_data.json",
                         num_val_im=0, filter_empty_rels=False, filter_non_overlap=False,
                         filter_duplicate_rels=False)
print(len(vg_test_dataset))
coco_ids_in_vg_test = [x['coco_id'] for x in vg_test_dataset.img_info if x['coco_id'] is not None]

########################### caption data process ############################################

processed_coco_caption_file = 'dataset/VG/aligned_triplet_info_vg.json'
print(f"Process_File:{processed_coco_caption_file.split('/')[-1]}")

with open('dataset/COCO/captions_train2017.json', 'r') as fin:
    coco_captions = json.load(fin) # original caption annotation

with open('dataset/COCO/captions_val2017.json', 'r') as fin:
    coco_captions_val = json.load(fin) # original caption annotation

id_64K_list = np.load("dataset/COCO/COCO_triplet_labels.npy", allow_pickle=True).tolist()['img_id_list'] # 64K image id list (SGNLS. Zhong et al. ICCV'21)

if os.path.exists(processed_coco_caption_file):
    with open(processed_coco_caption_file, 'r') as fin:
        coco_parsing_info = json.load(fin)

else:
    # Conventional approach (Parser+KB)
    coco_imgid2triplets = {}
    caption_object_vocabs, caption_relation_vocabs = set(), set()

    for cap in tqdm(coco_captions['annotations']):
        image_id = cap['image_id']
        
        if image_id in coco_ids_in_vg_test: continue
        if str(image_id) not in id_64K_list: continue

        cap_graph = sng_parser.parse(cap['caption'])
        for e in cap_graph['entities']:
            caption_object_vocabs.add(e['lemma_head'])
        for r in cap_graph['relations']:
            triplet = (cap_graph['entities'][r['subject']]['lemma_head'], r['lemma_relation'], cap_graph['entities'][r['object']]['lemma_head'])
            caption_relation_vocabs.add(r['lemma_relation'])

            if image_id not in coco_imgid2triplets:
                coco_imgid2triplets[image_id] = set()
            coco_imgid2triplets[image_id].add(triplet)


    for cap in tqdm(coco_captions_val['annotations']):
        image_id = cap['image_id']
        
        if image_id in coco_ids_in_vg_test: continue
        if str(image_id) not in id_64K_list: continue
        
        cap_graph = sng_parser.parse(cap['caption'])
        for e in cap_graph['entities']:
            caption_object_vocabs.add(e['lemma_head'])
        for r in cap_graph['relations']:
            triplet = (cap_graph['entities'][r['subject']]['lemma_head'], r['lemma_relation'], cap_graph['entities'][r['object']]['lemma_head'])
            caption_relation_vocabs.add(r['lemma_relation'])

            if image_id not in coco_imgid2triplets:
                coco_imgid2triplets[image_id] = set()
            coco_imgid2triplets[image_id].add(triplet)


    # map caption objects/predicates to VG150 categories
    caption_object_vocabs_to_vg_objs, caption_relation_vocabs_to_vg_rels = map_caption_concepts_to_vg150_categories(
        list(caption_object_vocabs), list(caption_relation_vocabs), vg_test_dataset
    )

    # convert to and only keep VG150 valid triplets
    image_sg_infos = {}
    for imgid, triplets in coco_imgid2triplets.items():
        vg150_valid_objects, vg150_valid_triplets = set(), set()
        for triplet in triplets:
            subjs = caption_object_vocabs_to_vg_objs[triplet[0]]
            preds = caption_relation_vocabs_to_vg_rels[triplet[1]]
            objs = caption_object_vocabs_to_vg_objs[triplet[2]]
            if len(subjs) > 0:
                subj = random.choice(subjs); vg150_valid_objects.add(subj)
                if len(objs) > 0:
                    obj = random.choice(objs); vg150_valid_objects.add(obj)
                    if len(preds) > 0:
                        vg150_valid_triplets.add((subj, random.choice(preds), obj))

        image_sg_infos[imgid] = {
            'objects': list(vg150_valid_objects),
            'triplets': list(vg150_valid_triplets),
            'original_triplets': list(triplets)
        }

    coco_parsing_info = {
        'text_scene_graph': image_sg_infos,
        'caption_object_vocabs_to_vg_objs': caption_object_vocabs_to_vg_objs,
        'caption_relation_vocabs_to_vg_rels': caption_relation_vocabs_to_vg_rels
    }
    with open(processed_coco_caption_file, 'w', encoding='utf-8') as fout:
        json.dump(coco_parsing_info, fout, ensure_ascii=False, indent=4)


########################### obtain grounded boxes of entities ###########################
if 'sg_grounding_infos' not in coco_parsing_info:
    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.,
        show_mask_heatmaps=False
    )
    grounded_img_ids = []
    all_sg_box_count, grounded_sg_box_count = 0, 0
    coco_imgid2filename = {x['id']: x['file_name'] for x in coco_captions['images']}
    coco_imgid2filename_val = {x['id']: x['file_name'] for x in coco_captions_val['images']}
    coco_imgid2filename.update(coco_imgid2filename_val)
    for index, (img_id, img_sg_info) in enumerate(tqdm(coco_parsing_info['text_scene_graph'].items())):
        if index % args.gpu_size != args.local_rank or len(img_sg_info['triplets']) < 1: continue
        grounded_img_ids.append(img_id)
        img_filename = f"dataset/COCO/images/{coco_imgid2filename[int(img_id)]}"
        pil_image = Image.open(img_filename).convert("RGB")
        input_image = np.array(pil_image)[:, :, [2, 1, 0]]

        # caption from parsed text scene graph
        img_pseudo_caption = ''
        entities, tokens_positive = [], []
        img_triplets = []
        for (s, p, o) in img_sg_info['triplets']:
            triplet_caption = f"{s} {p} {o}. "
            img_pseudo_caption += triplet_caption
            entities.append(s); tokens_positive.append([[len(img_pseudo_caption)-len(triplet_caption), len(img_pseudo_caption)-len(triplet_caption)+len(s)]])
            entities.append(o); tokens_positive.append([[len(img_pseudo_caption)-2-len(o), len(img_pseudo_caption)-2]])
            img_triplets.append({
                "subject": len(entities)-2,
                "object": len(entities)-1,
                "vg150_predicate_label": vg_test_dataset.ind_to_predicates.index(p),
                "text": (s, p, o)
            })

        # grounding
        glip_demo.entities = entities
        glip_demo.entities_caption_span = tokens_positive
        _, box_predictions = glip_demo.run_on_web_image(input_image, img_pseudo_caption, 0., show_result=False, custom_entity=tokens_positive)
        # imshow(_, img_pseudo_caption)

        # collect entities and relations
        w, h = box_predictions.size
        ent_boxes = torch.zeros((len(entities), 4))
        ent_boxes[:, 2] = w-1; ent_boxes[:, 3] = h-1

        visual_boxes = box_predictions.bbox
        visual_boxes_ent_ids = box_predictions.get_field('labels')-1 # ent_id = label_id-1
        visual_boxes_scores = box_predictions.get_field('scores')
        entities_names2ids = {}
        for ent_id in visual_boxes_ent_ids.unique(): # grounded box
            s, ind = visual_boxes_scores[visual_boxes_ent_ids == ent_id].topk(1)
            ent_box = visual_boxes[visual_boxes_ent_ids == ent_id][ind[0]]
            ent_boxes[ent_id] = ent_box

            label_name = entities[ent_id]
            if label_name in entities_names2ids:
                entities_names2ids[label_name].append(ent_id)
            else:
                entities_names2ids[label_name] = [ent_id]

        img_entities = []
        for ent_id in range(len(entities)):
            label_name = entities[ent_id]
            if (ent_id not in visual_boxes_ent_ids.unique()) and label_name in entities_names2ids:
                same_name_ent_id = random.choice(entities_names2ids[label_name])
                ent_boxes[ent_id] = ent_boxes[same_name_ent_id]

            img_entities.append({
                'label_name': label_name,
                "vg150_obj_label": vg_test_dataset.ind_to_classes.index(label_name),
                'xyxy_box': ent_boxes[ent_id].tolist()
            })

        img_sg_info['entities_after_grounding'] = img_entities
        img_sg_info['relations_after_grounding'] = img_triplets
        all_sg_box_count += len(img_entities)
        grounded_sg_box_count += len(visual_boxes_ent_ids.unique())

    coco_parsing_info['sg_grounding_infos'] = {
        'all_sg_box_count': all_sg_box_count,
        'grounded_sg_box_count': grounded_sg_box_count,
        'grounded_img_ids': grounded_img_ids
    }
    with open(f"{processed_coco_caption_file.replace('.json', '')}_grounded.json", 'w', encoding='utf-8') as fout:
        json.dump(coco_parsing_info, fout, ensure_ascii=False, indent=4)

sg_grounding_infos = coco_parsing_info['sg_grounding_infos']
print(f"\nall_sg_box_count={sg_grounding_infos['all_sg_box_count']}")
print(f"grounded_sg_box_count={sg_grounding_infos['grounded_sg_box_count']}")
