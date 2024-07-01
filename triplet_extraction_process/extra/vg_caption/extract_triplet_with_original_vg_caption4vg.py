import os
import sys
import json
import openai
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
api_key = str(sys.argv[1])
openai.api_key = api_key


vg_description = json.load(open("dataset/VG_Caption/region_descriptions.json", 'r'))

with open("dataset/VG_Caption/img_id_list.pkl", 'rb') as f:
    img_id_list = pickle.load(f)

caption_dict = defaultdict(list)
total_cnt = 0
for vg in vg_description:
    img_id = vg['id']
    if img_id not in img_id_list: continue
    region_caption = vg['regions']
    for r in region_caption:
        caption_dict[img_id].append(r['phrase'])
        total_cnt += 1
print(f"Total: {total_cnt} / Set Count: {len(caption_dict)}")


number_dict = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'tweleve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen'}

# In-contenxt Few-shot learning
def extract_triplets(captions):
    caption_list = []
    for c in captions:
        caption_list.append(c.strip('\n').strip().strip('"').strip("'").strip("."))
    question = f"\
    From the given sentence, the task is to extract meaningful triplets formed as <subject, predicate, object>. Note that the subject is the entity or noun that performs the action or is being described, and the object is the entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being). \n\
    Let's take a few examples to understand how to extract meaningful triplets. \
    Question: Given the sentence 'the woman infront of the computer has black hair,' extract meaningful triplets. Answer: \
    Meaningful triplets are <woman, in front of, computer> and <woman, has, hair>. \n\
    Question: Given the sentence 'several trees growing on the hillside,' extract meaningful triplets. Answer: \
    A meaningful triplet is <tree, growing on, hillsde>. \n\
    Question: Given the sentence 'a black bike parked on the side of the road,' extract meaningful triplets. Answer: \
    A meaningful triplet is <bike, parked on, road>. \n\
    Question: Given the sentence 'cables lying on floor in front of TV,' extract meaningful triplets. Answer: \
    Meaningful triplets are <cable, lying on, floor> and <cable, in front of, TV>. \n\
    Question: Given the sentence 'An airplane with a blue tail that is flying in the sky,' extract meaningful triplets. Answer: \
    Meaningful triplets are <airplane, with, tail> and <airplane, flying in, sky>. \n\
    Question: Given the sentence 'two people sitting on a green bench,' extract meaningful triplets. Answer: \
    A meaningful triplet is <people, sitting on, bench>. \n\
    Question: Given the sentence 'Green roof on top of the building made of brick,' extract meaningful triplets. Answer: \
    Meaningful triplets are <roof, on, building> and <building, made of, brick>. \n\
    Please answer the following {len(caption_list)} questions. \
    "
    for c in caption_list:
        question += f"Question: Given the sentence '{c}', extract meaningful triplets. Answer: "
    
    completion = openai.ChatCompletion.create(
    model='gpt-3.5-turbo', 
    messages=[{"role": "user", "content": question}],
    temperature=0
    )
    response = completion.choices[0].message.content
    return response, refine_output(response)

def refine_output(response):
    triplet_list = []
    for i, res in enumerate(response.split("Question")):
        if i == 0: continue
        triplets = res.split("Answer:")[1]
        for a in triplets[triplets.find("triplet"):].strip("\n").strip('\n').strip('.').split("<")[1:]:
            a = a.split(",")
            if len(a) < 3: continue
            sub, pred, obj = a[0].strip(), a[1].strip(), a[2].split(">")[0].strip()
            if obj == '': continue
            triplet_list.append([sub, pred, obj])
    return triplet_list

# Extract triplet
total = len(img_id_list)
start = 0; end = len(img_id_list)
save_period = 50
print(f"Start: {start}, End: {end}, Total: {total} - Original")
triplet_info = {}
filter_id = []
gpt_count = 0
error_count = 0

for iter, i in tqdm(enumerate(img_id_list[start:end])):
    captions_list = caption_dict[int(i)]
    triplet_info[i] = {}
    triplet_info[i]['before_matched_triplets'] = []
    for n in np.arange(0, len(captions_list), 10):
        part_caption_list = captions_list[n: n+10]
        try:
            response, triplet = extract_triplets(part_caption_list)
            triplet_info[i]['before_matched_triplets'].extend(triplet)
        except:
            error_count += 1
            print("Error")
            continue

    if iter % save_period == 0:
        previous_path = f"dataset/VG_Caption/start_{start}_end_{end}_{iter-save_period}_original.pkl"
        if os.path.isfile(previous_path):
            os.remove(previous_path)
            
        with open(f"dataset/VG_Caption/start_{start}_end_{end}_{iter}_original.pkl", 'wb') as f:
            pickle.dump(triplet_info, f)
            

  
    if iter == end-start-1:
        previous_path = f"dataset/VG_Caption/start_{start}_end_{end}_{end-start-save_period}_original.pkl"
        if os.path.isfile(previous_path):
            os.remove(previous_path)
            
        with open(f"dataset/VG_Caption/start_{start}_end_{end}_final_original.pkl", 'wb') as f:
            pickle.dump(triplet_info, f)     
           

