import os
import re
import sys
import json
import pickle
import openai
import numpy as np
from tqdm import tqdm
from collections import defaultdict
api_key = str(sys.argv[1])
openai.api_key = api_key

# VG Information
predicate_info = json.load(open("dataset/VG/VG-SGG-dicts-with-attri.json", 'r'))
entity_set = list(predicate_info['idx_to_label'].values())
predicate_set = list(predicate_info['idx_to_predicate'].values())

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

def contains_characters(input_string):
    alphabetic_characters = re.findall(r'[a-zA-Z]', input_string)
    alphabetic_string = ''.join(alphabetic_characters)
    return alphabetic_string


# Find relevant lexemes - relation
def alignment_relation(relation):
    question = f"\
    The predefined predicate lexicon containing 50 lexemes is numbered as follows: 1.above 2.across 3.against 4.along 5.and 6.at 7.attached to 8.behind 9.belonging to 10.between 11.carrying 12.covered in 13.covering 14.eating 15.flying in 16.for 17.from 18.growing on 19.hanging from 20.has 21.holding 22.in 23.in front of 24.laying on 25.looking at 26.lying on 27.made of 28.mounted on 29.near 30.of 31.on 32.on back of 33.over 34.painted on 35.parked on 36.part of 37.playing 38.riding 39.says 40.sitting on 41.standing on 42.to 43.under 44.using 45.walking in 46.walking on 47.watching 48.wearing 49.wears 50.with. \n\
    Given the lexeme, the task is to find semantically relevant lexeme from the predefined predicate lexicon. However, if there is no semantically relevant lexeme in the predefined predicate lexicon, please answer 0.None. \
    Let's take a few examples. \n\
    Question: Given the lexeme 'next to,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 29.near \
    Question: Given the lexeme 'uses,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 44.using \
    Question: Given the lexeme 'reading,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 25.looking at \
    Question: Given the lexeme 'sitting at,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 40.sitting on \
    Question: Given the lexeme 'grazing,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 14.eating \
    Question: Given the lexeme 'pointing to,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 0.None \
    Question: Given the lexeme 'lies on,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 24.lying on \
    Question: Given the lexeme 'covered with,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 12.covered in \
    Question: Given the lexeme 'placed next to,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 29.near \
    Question: Given the lexeme 'looking down at,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 25.looking at \
    Question: Given the lexeme 'containing,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 0.has \
    Question: Given the lexeme 'playing with,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 37.playing \
    Question: Given the lexeme 'driving,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 0.None \
    Question: Given the lexeme 'holds,' find semantically relevant lexeme in the predefined predicate lexicon. Answer: 21.holding \
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
    The predefined entity lexicon containing 150 lexemes is numbered as follows: 1.airplane 2.animal 3.arm 4.bag 5.banana 6.basket 7.beach 8.bear 9.bed 10.bench 11.bike 12.bird 13.board 14.boat 15.book 16.boot 17.bottle 18.bowl 19.box 20.boy 21.branch 22.building 23.bus 24.cabinet 25.cap 26.car 27.cat 28.chair 29.child 30.clock 31.coat 32.counter 33.cow 34.cup 35.curtain 36.desk 37.dog 38.door 39.drawer 40.ear 41.elephant 42.engine 43.eye 44.face 45.fence 46.finger 47.flag 48.flower 49.food 50.fork 51.fruit 52.giraffe 53.girl 54.glass 55.glove 56.guy 57.hair 58.hand 59.handle 60.hat 61.head 62.helmet 63.hill 64.horse 65.house 66.jacket 67.jean 68.kid 69.kite 70.lady 71.lamp 72.laptop 73.leaf 74.leg 75.letter 76.light 77.logo 78.man 79.men 80.motorcycle 81.mountain 82.mouth 83.neck 84.nose 85.number 86.orange 87.pant 88.paper 89.paw 90.people 91.person 92.phone 93.pillow 94.pizza 95.plane 96.plant 97.plate 98.player 99.pole 100.post 101.pot 102.racket 103.railing 104.rock 105.roof 106.room 107.screen 108.seat 109.sheep 110.shelf 111.shirt 112.shoe 113.short 114.sidewalk 115.sign 116.sink 117.skateboard 118.ski 119.skier 120.sneaker 121.snow 122.sock 123.stand 124.street 125.surfboard 126.table 127.tail 128.tie 129.tile 130.tire 131.toilet 132.towel 133.tower 134.track 135.train 136.tree 137.truck 138.trunk 139.umbrella 140.vase 141.vegetable 142.vehicle 143.wave 144.wheel 145.window 146.windshield 147.wing 148.wire 149.woman 150.zebra. \n\
    Given the lexeme, the task is to find semantically relevant lexeme from the predefined entity lexicon. However, if there is no semantically relevant lexeme in the predefined entity lexicon, please answer 0.None. \
    Let's take a few examples. \n\
    Question: Given the lexeme 'water,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 0.None \
    Question: Given the lexeme 'smartphone,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 92.phone \
    Question: Given the lexeme 'steel,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 0.None \
    Question: Given the lexeme 'sea,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 143.wave \
    Question: Given the lexeme 'city,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 22.building \
    Question: Given the lexeme 'cobble,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 104.rock \
    Question: Given the lexeme 'flowers,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 48.flower \
    Question: Given the lexeme 't-shirt,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 111.shirt \
    Question: Given the lexeme 'blue,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 0.None \
    Question: Given the lexeme 'boys,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 20.boy \
    Question: Given the lexeme 'she,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 149.woman \
    Question: Given the lexeme 'pigeon,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 12.bird \
    Question: Given the lexeme 'grass,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 96.plant \
    Question: Given the lexeme 'trees,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 136.tree \
    Question: Given the lexeme 'couple,' find semantically relevant lexeme in the predefined entity lexicon. Answer: 90.people \
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

original_triplet_dict = json.load(open("dataset/VG_Caption/misaligned_triplets_vg_caption.json", 'r'))


# Save the misaligned entity & predicate set
misaligned_entity = set()
misaligned_relation = set()
pred_num_dict = defaultdict(int)
ent_num_dict = defaultdict(int)
cnt = 0
for i in list(original_triplet_dict.keys()):
    triplet_set = set()
    for v1 in original_triplet_dict[i]:
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
    v = v.lower()
    # Filter the error (number is in the entity)
    if v.strip().isdigit():
        aligned_entity_dict[v] = '0.None'
    elif len(contains_characters(v)) <= 1:
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
            temp_list.append(extra_info[j].strip("'").strip('"').lower())
        else:
            temp_list.append(j.strip("'").strip('"').lower())
    try:
        response = alignment_entity(temp_list)
    except:
        continue
    for ent, res in zip(misaligned_entity[i:i+10], response.split("Answer:")[1:]):
        aligned_entity_dict[ent] = res[:res.find('\n')].strip()

with open("dataset/VG_Caption/aligned_entity_dict_vg_caption4vg.pkl", 'wb') as f:
    pickle.dump(aligned_entity_dict, f)
    
    
## Predicate
extra_info = {}
for r in misaligned_relation:
    if len(r.split(" ")) > 4:
        aligned_relation_dict[r] = '0.None'
    elif len(contains_characters(r)) <= 1:
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

with open("dataset/VG_Caption/aligned_predicate_dict_vg_caption4vg.pkl", 'wb') as f:
    pickle.dump(aligned_relation_dict, f)




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

## Predicate
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


# Save
with open("dataset/VG_Caption/aligned_entity_dict_vg_caption4vg.pkl", 'wb') as f:
    pickle.dump(aligned_entity_dict, f)

with open("dataset/VG_Caption/aligned_predicate_dict_vg_caption4vg.pkl", 'wb') as f:
    pickle.dump(aligned_relation_dict, f)