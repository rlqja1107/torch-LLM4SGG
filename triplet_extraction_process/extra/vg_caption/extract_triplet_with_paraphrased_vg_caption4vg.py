import os
import sys
import json
import openai
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
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
    From the given sentence, the task is to extract meaningful triplets formed as <subject, predicate, object>.\
    To extract meaningful triplets from the sentence, please follow the following two steps. \
    Step 1: Paraphrase the sentence. \n\
    Step 2: From the paraphrased sentence obtained in the Step 1, extract meaningful triplets formed as <subject, predicate, object>. Note that the subject is the entity or noun that performs the action or is being described, and the object is the entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being). \n\
    Let's take a few examples to understand how to extract meaningful triplets. \
    Question: Given the sentence 'A man dressed in a red shirt and black pants,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'A man wearing a red shirt and black pants'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence are: <man, wearing, shirt>, <man, wearing, pant>. \
    Output: <man, wearing, shirt>, <man, wearing, pant>. \n\
    Question: Given the sentence 'The woman's gray shirt,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'The gray shirt belonging to the women'. \
    Step 2: A meaningful triplet, where the subject and object are the simple noun, extracted from the paraphrased sentence is: <shirt, belonging to, women>. \
    Output: <shirt, belonging to, women>. \n\
    Question: Given the sentence 'White shorts on a man holding a tennis racket,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'A man, holding a tennis racket, is wearing white shorts.'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <man, holding, racket>, <man, wearing, short>. \
    Output: <man, holding, racket>, <man, wearing, short>. \n\
    Question: Given the sentence 'building across street from ramp,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'A building located on the opposite side of the ramp'. \
    Step 2: A meaningful triplet, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 is: <building, located on, ramp>. \
    Output: <building, located on, ramp>. \n\
    Question: Given the sentence 'four soldiers on horses,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'Four cavalrymen riding on horses'. \
    Step 2: A meaningful triplet, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 is: <cavalrymen, riding on, horse>. \
    Output: <cavalrymen, riding on, horse>. \n\
    Question: Given the sentence 'A black strap connected to a backpack,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'Black strap attached to backpack'. \
    Step 2: A meaningful triplet, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 is: <strap, attached to, backpack>. \
    Output: <strap, attached to, backpack>. \n\
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
    paraphrased_caption = []
    for i, res in enumerate(response.split("Question")):
        if i == 0: continue
        p_sentence = res.split(":")[4]
        triplets = res.split("Output:")[1].strip('\n').strip('\n').strip().strip('.')
        p_sentence = p_sentence[p_sentence.find("'")+1: p_sentence.rfind("'")].strip().strip('').strip(".")
        paraphrased_caption.append(p_sentence)
        for a in triplets.split("<")[1:]:
            a = a.split(",")
            if len(a) < 3: continue
            sub, pred, obj = a[0].strip(), a[1].strip(), a[2].split(">")[0].strip().strip(".")
            triplet_list.append([sub, pred, obj])
    return triplet_list, paraphrased_caption

# In-context learning to generate triplet
total = len(img_id_list)
start = 0; end = len(img_id_list)
save_period = 50
print(f"Start: {start}, End: {end}, Total: {total}, Save period: {save_period} - Paraphrase")
triplet_info = {}
filter_id = []
gpt_count = 0
error_count = 0
for iter, i in tqdm(enumerate(img_id_list[start:end])):
    captions_list = caption_dict[int(i)]
    triplet_info[i] = {}
    triplet_info[i]['p_sentence'] = []
    triplet_info[i]['before_matched_triplets'] = []
    for n in np.arange(0, len(captions_list), 5):
        part_caption_list = captions_list[n: n+5]
        try:
            response, (triplet, paraphrased_cap) = extract_triplets(part_caption_list)
            triplet_info[i]['before_matched_triplets'].extend(triplet)
            triplet_info[i]['p_sentence'].extend(paraphrased_cap)
        except:
            error_count += 1
            print("Error")
            continue

        
    if iter % save_period == 0:
        previous_path = f"dataset/VG_Caption/start_{start}_end_{end}_{iter-save_period}_paraphrase.pkl"
        if os.path.isfile(previous_path):
            os.remove(previous_path)
            
        with open(f"dataset/VG_Caption/start_{start}_end_{end}_{iter}_paraphrase.pkl", 'wb') as f:
            pickle.dump(triplet_info, f)
            
  
    if iter == end-start-1:
        previous_path = f"dataset/VG_Caption/start_{start}_end_{end}_{end-start-save_period}_paraphrase.pkl"
        if os.path.isfile(previous_path):
            os.remove(previous_path)
            
        with open(f"dataset/VG_Caption/start_{start}_end_{end}_final_paraphrase.pkl", 'wb') as f:
            pickle.dump(triplet_info, f)     