import os
import re
import sys
import json
import pickle
import openai
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
api_key = str(sys.argv[1])
openai.api_key = api_key


# VG Information
predicate_info = json.load(open("dataset/GQA/GQA_200_ID_Info.json", 'r'))
entity_set = predicate_info['ind_to_classes']
del entity_set[0] # background
predicate_set = predicate_info['ind_to_predicates']
del predicate_set[0] # background

entity_to_idx = {}
for i,ent in enumerate(entity_set):
    entity_to_idx[ent] = i+1
predicate_to_idx = {}
for i,pred in enumerate(predicate_set):
    predicate_to_idx[pred] = i+1
    
    
def remove_digits_from_string(input_string):
    # Define a regular expression pattern to match digits (\d+)
    pattern = r'\d+'
    
    # Use re.sub() to replace digits with an empty string
    cleaned_string = re.sub(pattern, '', input_string)
    
    return cleaned_string

def detect_digits_in_string(input_string):
    # Define a regular expression pattern to match digits (\d+)
    pattern = r'\d+'
    
    # Use re.findall() to find all occurrences of the pattern in the string
    digit_matches = re.findall(pattern, input_string)
    
    return digit_matches


# Find relevant lexemes - relation
def alignment_relation(relation):
    question = f"\
    The predefined predicate lexicon containing 100 lexemes is numbered as follows: 1.on 2.wearing 3.of 4.near 5.in 6.behind 7.in front of 8.holding 9.next to 10.above 11.on top of 12.below 13.by 14.with 15.sitting on 16.on the side of 17.under 18.riding 19.standing on 20.beside 21.carrying 22.walking on 23.standing in 24.lying on 25.eating 26.covered by 27.looking at 28.hanging on 29.at 30.covering 31.on the front of 32.around 33.sitting in 34.parked on 35.watching 36.flying in 37.hanging from 38.using 39.sitting at 40.covered in 41.crossing 42.standing next to 43.playing with 44.walking in 45.on the back of 46.reflected in 47.flying 48.touching 49.surrounded by 50.covered with 51.standing by 52.driving on 53.leaning on 54.lying in 55.swinging 56.full of 57.talking on 58.walking down 59.throwing 60.surrounding 61.standing near 62.standing behind 63.hitting 64.printed on 65.filled with 66.catching 67.growing on 68.grazing on 69.mounted on 70.facing 71.leaning against 72.cutting 73.growing in 74.floating in 75.driving 76.beneath 77.contain 78.resting on 79.worn on 80.walking with 81.driving down 82.on the bottom of 83.playing on 84.playing in 85.feeding 86.standing in front of 87.waiting for 88.running on 89.close to 90.sitting next to 91.swimming in 92.talking to 93.grazing in 94.pulling 95.pulled by 96.reaching for 97.attached to 98.skiing on 99.parked along 100.hang on. \n\
    Given the lexeme, the task is to find semantically relevant lexeme from the predefined predicate lexicon. However, if there is no semantically relevant lexeme in the predefined predicate lexicon, please answer 0.None. \
    Let's take a few examples. \n\
    Question: Given the lexeme 'seated in', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 15.sitting on \
    Question: Given the lexeme 'are parked in', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 34.parked on \
    Question: Given the lexeme 'contains', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 77.contain \
    Question: Given the lexeme 'gripping', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 94.pulling \
    Question: Given the lexeme 'skiing down', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 98.skiing on \
    Question: Given the lexeme 'smiling', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 0.None \
    Question: Given the lexeme 'placed next to', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 9.next to \
    Question: Given the lexeme 'sleeping on', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 24.lying on \
    Question: Given the lexeme 'reading', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 35.watching \
    Question: Given the lexeme 'crowded with', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 0.None \
    Question: Given the lexeme 'perched on', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 15.sitting on \
    Question: Given the lexeme 'posted on', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 97.attached to \
    Question: Given the lexeme 'placed on top of', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 11.on top of \
    Question: Given the lexeme 'cycling', find semantically relevant lexeme in the predefined predicate lexicon. Answer: 18.riding \
    Please answer the following {len(relation)} questions. \
    "
    for r in relation:
        question += f"Question: Given the lexeme '{r}', find semantically relevant lexeme in the predefined predicate lexicon. Answer:"
    completion = openai.ChatCompletion.create(
    model='gpt-3.5-turbo', 
    messages=[{"role": "user", "content": question}],
    temperature=0
    )
    response = completion.choices[0].message.content
    return response

# Find relevant lexemes - entity
def alignment_entity(entity_list):
    question = f"\
    The predefined entity lexicon containing 200 lexemes is numbered as follows: 1.window 2.tree 3.man 4.shirt 5.wall 6.building 7.person 8.ground 9.sky 10.leg 11.sign 12.hand 13.head 14.pole 15.grass 16.hair 17.car 18.ear 19.eye 20.woman 21.clouds 22.shoe 23.table 24.leaves 25.wheel 26.door 27.pants 28.letter 29.people 30.flower 31.water 32.glass 33.chair 34.fence 35.arm 36.nose 37.number 38.floor 39.rock 40.jacket 41.hat 42.plate 43.tail 44.leaf 45.face 46.bush 47.shorts 48.road 49.bag 50.sidewalk 51.tire 52.helmet 53.snow 54.boy 55.umbrella 56.logo 57.roof 58.boat 59.bottle 60.street 61.plant 62.foot 63.branch 64.post 65.jeans 66.mouth 67.cap 68.girl 69.bird 70.banana 71.box 72.bench 73.mirror 74.picture 75.pillow 76.book 77.field 78.glove 79.clock 80.dirt 81.bowl 82.bus 83.neck 84.trunk 85.wing 86.horse 87.food 88.train 89.kite 90.paper 91.shelf 92.airplane 93.sock 94.house 95.elephant 96.lamp 97.coat 98.cup 99.cabinet 100.street light 101.cow 102.word 103.dog 104.finger 105.giraffe 106.mountain 107.wire 108.flag 109.seat 110.sheep 111.counter 112.skis 113.zebra 114.hill 115.truck 116.bike 117.racket 118.ball 119.skateboard 120.ceiling 121.motorcycle 122.player 123.surfboard 124.sand 125.towel 126.frame 127.container 128.paw 129.feet 130.curtain 131.windshield 132.traffic light 133.horn 134.cat 135.child 136.bed 137.sink 138.animal 139.donut 140.stone 141.tie 142.pizza 143.orange 144.sticker 145.apple 146.backpack 147.vase 148.basket 149.drawer 150.collar 151.lid 152.cord 153.phone 154.pot 155.vehicle 156.fruit 157.laptop 158.fork 159.uniform 160.bear 161.fur 162.license plate 163.lady 164.tomato 165.tag 166.mane 167.beach 168.tower 169.cone 170.cheese 171.wrist 172.napkin 173.toilet 174.desk 175.dress 176.cell phone 177.faucet 178.blanket 179.screen 180.watch 181.keyboard 182.arrow 183.sneakers 184.broccoli 185.bicycle 186.guy 187.knife 188.ocean 189.t-shirt 190.bread 191.spots 192.cake 193.air 194.sweater 195.room 196.couch 197.camera 198.frisbee 199.trash can 200.paint. \n\
    Given the lexeme, the task is to find semantically relevant lexeme from the predefined entity lexicon. However, if there is no semantically relevant lexeme in the predefined entity lexicon, please answer 0.None. \
    Let's take a few examples. \n\
    Question: Given the lexeme 'skier', find semantically relevant lexeme in the predefined entity lexicon. Answer: 7.person \
    Question: Given the lexeme 'bathroom', find semantically relevant lexeme in the predefined entity lexicon. Answer: 173.toilet \
    Question: Given the lexeme 'egg', find semantically relevant lexeme in the predefined entity lexicon. Answer: 87.food \
    Question: Given the lexeme 'vanity', find semantically relevant lexeme in the predefined entity lexicon. Answer: 91.shelf \
    Question: Given the lexeme 'desktop', find semantically relevant lexeme in the predefined entity lexicon. Answer: 157.laptop \
    Question: Given the lexeme 'cobble', find semantically relevant lexeme in the predefined entity lexicon. Answer: 39.rock \
    Question: Given the lexeme 'blue', find semantically relevant lexeme in the predefined entity lexicon. Answer: 0.None \
    Question: Given the lexeme 'poles', find semantically relevant lexeme in the predefined entity lexicon. Answer: 14.pole \
    Question: Given the lexeme 'motorcyclist', find semantically relevant lexeme in the predefined entity lexicon. Answer: 122.player \
    Question: Given the lexeme 'pigeon', find semantically relevant lexeme in the predefined entity lexicon. Answer: 69.bird \
    Question: Given the lexeme 'forest', find semantically relevant lexeme in the predefined entity lexicon. Answer: 106.mountain \
    Question: Given the lexeme 'bat', find semantically relevant lexeme in the predefined entity lexicon. Answer: 0.None \
    Question: Given the lexeme 'surfboards', find semantically relevant lexeme in the predefined entity lexicon. Answer: 123.surfboard \
    Question: Given the lexeme 'tray', find semantically relevant lexeme in the predefined entity lexicon. Answer: 42.plate \
    Question: Given the lexeme 'wastebasket', find semantically relevant lexeme in the predefined entity lexicon. Answer: 199.trash can \
    Please answer the following {len(entity_list)} questions. \
    "
    for r in entity_list:
        question += f"Question: Given the lexeme '{r}', find semantically relevant lexeme in the predefined entity lexicon. Answer:"
    completion = openai.ChatCompletion.create(
    model='gpt-3.5-turbo', 
    messages=[{"role": "user", "content": question}],
    temperature=0
    )
    response = completion.choices[0].message.content
    return response


# Bring the misaligned triplet information - original captions
with open("dataset/COCO/misaligned_triplets_original.pkl", 'rb') as f:
    misaligned_triplets_from_original = pickle.load(f)
    
# Bring the misaligned triplet information - paraphrased captions
with open("dataset/COCO/misaligned_triplets_paraphrased.pkl", 'rb') as f:
    misaligned_triplets_from_paraphrased = pickle.load(f)


# Save the misaligned entity & predicate set
misaligned_entity = set()
misaligned_relation = set()
pred_num_dict = defaultdict(int)
ent_num_dict = defaultdict(int)
cnt = 0
for i in list(misaligned_triplets_from_original.keys()):
    triplet_set = set()
    for k,v in misaligned_triplets_from_original[i]['original_triplet'].items():
        for v1 in v:
            sub, pred, obj = v1[0], v1[1], v1[2]
            misaligned_entity.add(sub)
            misaligned_entity.add(obj)
            misaligned_relation.add(pred)
            ent_num_dict[sub]+=1
            ent_num_dict[obj] += 1
            pred_num_dict[pred] += 1

for i in list(misaligned_triplets_from_paraphrased.keys()):
    triplet_set = set()
    for k,v in misaligned_triplets_from_paraphrased[i]['original_triplet'].items():
        for v1 in v:
            sub, pred, obj = v1[0], v1[1], v1[2]
            misaligned_entity.add(sub)
            misaligned_entity.add(obj)
            misaligned_relation.add(pred)
            ent_num_dict[sub]+=1
            ent_num_dict[obj] += 1
            pred_num_dict[pred] += 1           
            
misaligned_entity = list(misaligned_entity)
misaligned_relation = list(misaligned_relation)
ent_num_dict = dict(sorted(ent_num_dict.items(), key=lambda d: d[1], reverse=True))
pred_num_dict = dict(sorted(pred_num_dict.items(), key=lambda d:d[1], reverse=True))
print(f"# Misaligned Entity: {len(misaligned_entity)}, # Misaligned Relation: {len(misaligned_relation)}")

aligned_entity_dict = {}
aligned_relation_dict = {}
for i in entity_set:
    aligned_entity_dict[i] = f"{entity_to_idx[i]}.{i}"

for i in predicate_set:
    aligned_relation_dict[i] = f"{predicate_to_idx[i]}.{i}"

# Entity
# Pre-process the misaligned entity set
extra_info = {}
misaligned_entity = np.setdiff1d(misaligned_entity, list(aligned_entity_dict.keys()))
for v in misaligned_entity:
    # Filter the error (number is in the entity)
    if v.strip().isdigit():
        aligned_entity_dict[v] = '0.None'
    # Alignment using Root
    elif len(v.split(" ")) >= 2 and v.split(" ")[-1] in entity_set:
        aligned_entity_dict[v] = f"{entity_to_idx[v.split(' ')[-1]]}.{v.split(' ')[-1]}"
    # Only align the first element when 'and' is included
    elif "and" in v and len(v.split("and")) >= 2:
        extra_info[v] = v.split("and")[0].strip()
    elif "'" in v or '"' in v:
        extra_info[v] = v.strip("'").strip('"')

misaligned_entity = np.setdiff1d(misaligned_entity, list(aligned_entity_dict.keys()))
print(f"# Misaligned entity set: {len(misaligned_entity)}")
print()
print("Aligning the misaligned entity set")
for i in tqdm(range(0, len(misaligned_entity), 10)):
    temp_list = []
    for j in misaligned_entity[i:i+10]:
        if j in extra_info:
            temp_list.append(extra_info[j].strip("'").strip('"'))
        else:
            temp_list.append(j.strip("'").strip('"'))
    try:
        response = alignment_entity(temp_list)
    except:
        continue
    for ent, res in zip(misaligned_entity[i:i+10], response.split("Answer:")[1:]):
        aligned_entity_dict[ent] = res[:res.find('\n')].strip()

    
# Predicate
extra_info = {}
for r in misaligned_relation:
    if len(r.split(" ")) > 4:
        aligned_relation_dict[r] = '0.None'
        
# Pre-process the misaligned predicate set
misaligned_relation = np.setdiff1d(misaligned_relation, list(aligned_relation_dict.keys()))
for i in misaligned_relation:
    if len(i.split(" ")) > 1 and i.split(" ")[0] in ['is', 'for', 'being', 'are', 'can', 'arranged', 'were', 'was']:
        extra_info[i] = " ".join(i.split(" ")[1:]).strip()
    if len(i.split(" ")) > 1 and i.split(" ")[0] in ['can' ,'attempting', 'attempts', 'about'] and i.split(" ")[1] in ['be', 'to']:
        extra_info[i] = " ".join(i.split(" ")[2:]).strip()

print(f"Misaligned Predicate: {len(misaligned_relation)}")

print("Aligning the misaligned predicate set")
for i in tqdm(range(0, len(misaligned_relation), 10)):
    temp_list = []
    for j in misaligned_relation[i:i+10]:
        if j in extra_info:
            temp_list.append(extra_info[j].strip())
        else:
            temp_list.append(j)
    try:
        response = alignment_relation(temp_list)
    except:
        continue
    for ent, res in zip(misaligned_relation[i:i+10], response.split("Answer:")[1:]):
        aligned_relation_dict[ent] = res[:res.find('\n')].strip()



# Since LLM sometimes cause errors, pre-process the error.
# Entity
for k, v in aligned_entity_dict.items():
    if len(v.split(".")) ==2 and v.split(".")[1] in entity_set: continue
    
    if len(v.split(".")) ==2 and v.split(".")[1] in entity_set[int(v.split(".")[0])-1]:
        ent = entity_set[int(v.split(".")[0])-1]
        aligned_entity_dict[k] = f"{entity_to_idx[ent]}.{ent}"
        
    elif len(v.split(".")) < 2:
        if v in entity_set:
            aligned_entity_dict[k] = f"{entity_to_idx[v]}.{v}"
        else:
            aligned_entity_dict[k] = "0.None"
            
    elif v.split('.')[1] =='Non' or v.split('.')[1] =='No':
        aligned_entity_dict[k] = '0.None'
        
    elif ".".join(v.split('.')[1:]) not in entity_set and ".".join(v.split('.')[1:]) !='None':
        num = v.split(".")[0]
        if ".".join(v.split('.')[1:]) in entity_set:
            entity = entity_set[int(num)-1]
            aligned_entity_dict[k] = f"{num}.{entity}"
        else:
            aligned_entity_dict[k] = "0.None"

# Predicate
for k, v in aligned_relation_dict.items():
    if len(v.split(".")) ==2 and v.split(".")[1] in predicate_set:continue
    
    if len(v.split(".")) ==2 and v.split(".")[1] in predicate_set[int(v.split(".")[0])-1]:
        pred = predicate_set[int(v.split(".")[0])-1]
        aligned_relation_dict[k] = f"{predicate_to_idx[pred]}.{pred}"
        
    elif len(v.split(".")) < 2:
        aligned_relation_dict[k] = f"{predicate_to_idx[v]}.{v}"
        
    elif v.split('.')[1] =='Non' or v.split('.')[1] =='No':
        aligned_relation_dict[k] = '0.None'
        
    elif len(v.split(".")) >=2 and int(v.split(".")[0]) <= 50 and v.split(".")[1] in predicate_set[int(v.split(".")[0])-1]:
        pred = predicate_set[int(v.split(".")[0])-1]
        aligned_relation_dict[k] = f"{predicate_to_idx[pred]}.{pred}"
        
    elif ".".join(v.split('.')[1:]) not in predicate_set and ".".join(v.split('.')[1:]) !='None':
        num = v.split(".")[0]
        if ".".join(v.split('.')[1:]) in predicate_set:
            entity = predicate_set[int(num)-1]
            aligned_relation_dict[k] = f"{num}.{entity}"
        else:
            aligned_relation_dict[k] = "0.None"


with open("triplet_extraction_process/alignment_dict/aligned_entity_dict_gqa.pkl", 'wb') as f:
    pickle.dump(aligned_entity_dict, f)
    
with open("triplet_extraction_process/alignment_dict/aligned_predicate_dict_gqa.pkl", 'wb') as f:
    pickle.dump(aligned_relation_dict, f)


#for k, v in aligned_entity_dict.items():
#    if v.split(".")[1] !='None' and v.split(".")[1] not in entity_set:
#        print(k,v)
#for k, v in aligned_relation_dict.items():
#    if v.split(".")[1] !='None' and v.split(".")[1] not in predicate_set:
#        print(k,v)