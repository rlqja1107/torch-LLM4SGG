import sys
import json
import openai
import pickle
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict
api_key = str(sys.argv[1])
openai.api_key = api_key


engine = 'gpt-3.5-turbo'

# Match caption information
train2017_caption = json.load(open("dataset/COCO/captions_train2017.json", 'r'))
val2017_caption = json.load(open("dataset/COCO/captions_val2017.json", 'r'))

caption_info = defaultdict(list)
for k in train2017_caption['annotations']:
    caption_info[k['image_id']].append(k['caption'])
for k in val2017_caption['annotations']:
    caption_info[k['image_id']].append(k['caption'])
img_id_info_64K = np.load("dataset/COCO/COCO_triplet_labels.npy", allow_pickle=True).tolist()
img_id_list = img_id_info_64K['img_id_list']

number_dict = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven'}

# In-contenxt Few-shot learning
def extract_triplets(captions):
    caption_list = []
    for c in captions:
        caption_list.append(c.strip('\n').strip("").strip("'").strip("."))
    question = f"\
    From the given sentence, the task is to extract meaningful triplets formed as <subject, predicate, object>.\
    To extract meaningful triplets from the sentence, please follow the following two steps. \
    Step 1: Paraphrase the sentence. \n\
    Step 2: From the paraphrased sentence obtained in the Step 1, extract meaningful triplets formed as <subject, predicate, object>. Note that the subject is the entity or noun that performs the action or is being described, and the object is the entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being). \n\
    Let's take a few examples to understand how to extract meaningful triplets. \
    Question: Given the sentence 'a slice of bread is covered with a sour cream and quacamole,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'A piece of bread is topped with both sour cream and guacamole'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence are: <bread, topped with, sour cream>, <bread, topped with, guacamole>. \
    The meaningful triplets are <bread, topped with, sour cream>, and <bread, topped with, guacamole>. \n\
    Question: Given the sentence 'A beautiful woman walking a dog on top of a beach,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'A lovely woman strolling with a dog on the beach'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence are: <woman, strolling with, dog>, <woman, on, beach>, <dog, on, beach>. \
    The meaningful triplets are <woman, strolling with, dog>, <woman, on, beach>, and <dog, on, beach>. \n\
    Question: Given the sentence 'Four clock sitting on a floor next to a woman's feet,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'Four clocks are placed on the floor beside a woman's feet'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <clocks, placed on, floor>, <clocks, beside, feet>. \
    The meaningful triplets are <clocks, placed on, floor> and <clocks, beside, feet>. \n\
    Question: Given the sentence 'One person sits in a chair looking at her phone while another rests on the couch,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'A person is seated in a chair, using their phone, while someone else is relaxing on the couch'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <person, seated in, chair>, <person, using, phone>, <person, relaxing on, couch>. \
    The meaningful triplets are <person, seated in, chair>, <person, using, phone>, and <person, relaxing on, couch>. \n\
    Question: Given the sentence 'A lady and a child near a park bench with kites and ducks flying in the sky and on the ground,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'A woman and a child are close to a park bench, while kites soar through the sky and ducks move around on the ground'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <woman, close to, park bench>, <child, close to, park bench>, <kites, soar through, sky>, <ducks, move around, ground>. \
    The meaningful triplets are <woman, close to, park bench>, <child, close to, park bench>, <kites, soar through, sky>, and <ducks, move around, ground>. \n\
    Question: Given the sentence 'Two men sit on a bench near the sidewalk and one of them talks on a cell phone,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'Two guys are seated on a bench near the road, and one of them talks on a mobile phone'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <guys, seated on, bench>, <bench, near, road>, <guy, talks on, phone>. \
    The meaningful triplets are <guys, seated on, bench>, <bench, near, road>, and <guy, talks on, phone>. \n\
    Please answer the following {number_dict[len(caption_list)]} questions. \
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
    output_dict = {}
    for i, res in enumerate(response.split("Question")):
        if i == 0: continue
        output_dict[i] = {}
        p_sentence, triplets = res.split(":")[4], res.split(":")[6]
        p_sentence = p_sentence[p_sentence.find("'")+1: p_sentence.rfind("'")].strip().strip('').strip(".")
        before_match_parsing = []
        for a in triplets[triplets.find("triplet"):].split("<")[1:]:
            a = a.split(",")
            if len(a) < 3: continue
            sub, pred, obj = a[0].strip(), a[1].strip(), a[2].split(">")[0].strip()
            before_match_parsing.append([sub, pred, obj])
        output_dict[i]['p_sentence'] = p_sentence
        output_dict[i]['before_matched_triplets'] = before_match_parsing
    return output_dict

# In-context learning to generate triplet
start = 0; end = len(img_id_list)
print(f"Start: {start}, End: {end}")
triplet_info = {}
filter_id = []
gpt_count = 0
error_count = 0
for i in tqdm(img_id_list[start:end]):
    captions_list = caption_info[int(i)]
    try:
        _, output_dict = extract_triplets(captions_list)
    except:
        error_count += 1
        print("Error")
        continue
    triplet_info[i] = {}
    triplet_info[i]['parsed_caption'] = {}
    triplet_info[i]['original_triplet'] = {}
    for k, v in output_dict.items():
        triplet_info[i]['parsed_caption'][k] = v['p_sentence']
        triplet_info[i]['original_triplet'][k] = v['before_matched_triplets']
        
print(f"Error cnt: {error_count}")
with open(f"dataset/COCO/misaligned_triplets_paraphrased.pkl", 'wb') as f:
    pickle.dump(triplet_info, f)

