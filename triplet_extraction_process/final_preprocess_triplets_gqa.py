import json
import pickle
import numpy as np
from collections import defaultdict

# GQA Information
predicate_info = json.load(open("dataset/GQA/GQA_200_ID_Info.json", 'r'))
entity_set = predicate_info['ind_to_classes']
del entity_set[0]
predicate_set = predicate_info['ind_to_predicates']
del predicate_set[0]

with open(f"dataset/COCO/misaligned_triplets_original.pkl", 'rb') as f:
    triplet_info_original = pickle.load(f)

with open(f"dataset/COCO/misaligned_triplets_paraphrased.pkl", 'rb') as f:
    triplet_info_paraphrased = pickle.load(f)

with open("triplet_extraction_process/alignment_dict/aligned_entity_dict_gqa.pkl", 'rb') as f:
    aligned_entity_dict = pickle.load(f)
    
with open("triplet_extraction_process/alignment_dict/aligned_predicate_dict_gqa.pkl", 'rb') as f:
    aligned_relation_dict = pickle.load(f)
    

combined_triplet_info = defaultdict(set)

# Combine into one dict
for k, v in triplet_info_original.items():
    triplet_dict = v['original_triplet']
    for _, triplet_list in triplet_dict.items():
        for triplet in triplet_list:
            sub, pred, obj = triplet[0], triplet[1], triplet[2]
            sub = aligned_entity_dict[sub]; obj = aligned_entity_dict[obj]; pred = aligned_relation_dict[pred]
            sub = sub.split(".")[1]; obj = obj.split(".")[1]; pred = pred.split(".")[1]
            # Filter a triplet including None component
            if sub in entity_set and obj in entity_set and pred in predicate_set:
                combined_triplet_info[k].add((sub, pred, obj))  
        
for k, v in triplet_info_paraphrased.items():
    triplet_dict = v['original_triplet']
    for _, triplet_list in triplet_dict.items():
        for triplet in triplet_list:
            sub, pred, obj = triplet[0], triplet[1], triplet[2]
            sub = aligned_entity_dict[sub]; obj = aligned_entity_dict[obj]; pred = aligned_relation_dict[pred]
            sub = sub.split(".")[1]; obj = obj.split(".")[1]; pred = pred.split(".")[1]
            # Filter a triplet including None component
            if sub in entity_set and obj in entity_set and pred in predicate_set:
                combined_triplet_info[k].add((sub, pred, obj))  

# Choose fine-grained predicates
final_combined_triplet_info = {}
pred_count = defaultdict(int)

for k, v in combined_triplet_info.items():
    for triplet in v:
        pred_count[triplet[1]] += 1
pred_count = dict(sorted(pred_count.items(), key=lambda k:k[1], reverse=True))
pred_sorted_name = list(pred_count.keys())

for k, v in combined_triplet_info.items():
    per_img_triplet_dict = defaultdict(list)
    
    # Save multiple predicate per image
    for triplet in v:
        per_img_triplet_dict[(triplet[0], triplet[2])].append(triplet[1])
        
    triplet_set = []
    for pair, triplet_list in per_img_triplet_dict.items():
        sort_idx = []
        for pred in triplet_list:
            sort_idx.append(pred_sorted_name.index(pred))
        # Choose fine-grained predicate 
        fine_grained_pred = triplet_list[np.argmax(sort_idx)]
        triplet_set.append([pair[0], fine_grained_pred, pair[1]])
    final_combined_triplet_info[k] = triplet_set


# Transform the format into VS3's format
vs3_format_data = {}
vs3_format_data['text_scene_graph'] = {}
triplet_cnt = 0
img_cnt = 0
for img_id, triplets in final_combined_triplet_info.items():
    vs3_format_data['text_scene_graph'][img_id] = {}
    
    per_img_entity_set = set()
    per_img_triplet_set = set()
    for triplet in triplets:
        sub, pred, obj = triplet[0], triplet[1], triplet[2]
        per_img_entity_set.add(sub)
        per_img_entity_set.add(obj)
        per_img_triplet_set.add((sub, pred, obj))
    vs3_format_data['text_scene_graph'][img_id]['objects'] = list(per_img_entity_set)
    vs3_format_data['text_scene_graph'][img_id]['triplets'] = list(per_img_triplet_set)
    triplet_cnt += len(per_img_triplet_set)
    img_cnt += 1
    
#print(f"# Image: {img_cnt} / # Triplet: {triplet_cnt}")

with open("dataset/VG/aligned_triplet_info_gqa.json", 'w', encoding='utf-8') as f:
    json.dump(vs3_format_data, f, ensure_ascii=False, indent=4)