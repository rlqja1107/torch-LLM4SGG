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

# Please download below two files
data = pd.read_csv("dataset/CC/Train_GCC-training.tsv", delimiter='\t', keep_default_na=False, header=None)
triplet_label = np.load("dataset/CC/cc_triplet_labels.npy", allow_pickle=True).tolist()

img_id_list = []
for i in triplet_label['img_id_list']:
    img_id_list.append(int(i))


caption_dict = {}
for i in data.iloc[img_id_list].index:
    per_img_info = data.iloc[i]
    caption_dict[i] = {}
    caption_dict[i]['caption'] = per_img_info[0]
    caption_dict[i]['url'] = per_img_info[1]
    

number_dict = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven'}

# In-contenxt Few-shot learning
def extract_triplets(captions):
    captions = captions.strip('\n').strip().strip('"').strip("'").strip(".")
    question = f"\
    From the given sentence, the task is to extract meaningful triplets formed as <subject, predicate, object>. Note that the subject is the entity or noun that performs the action or is being described, and the object is the entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being). \n\
    Let's take a few examples to understand how to extract meaningful triplets. \
    Question: Given the sentence 'lone sad dog lying on the street,' extract meaningful triplets. Answer: \
    A meaningful triplet is <dog, lying on, street>. \n\
    Question: Given the sentence 'a woman sits on a surfboard in the ocean with a blue sky,' extract meaningful triplets. Answer: \
    Meaningful triplets are <woman, sits on, surfboard>, <woman, in, ocean>, and <ocean, with, sky>. \n\
    Question: Given the sentence 'a woman and her son walking along the tracks of a disused railway,' extract meaningful triplets. Answer: \
    Meaningful triplets are <woman, walking along, tracks>, <son, walking along, tracks>, and <tracks, of, railway>. \n\
    Question: Given the sentence 'a smiling man seated on a motorcycle that is parked on the sidewalk in front of a retail store,' extract meaningful triplets. Answer: \
    Meaningful triplets are <man, seated on, motorcycle>, <motorcycle, parked on, sidewalk>, and <motorcycle, in front of, store>. \n\
    Question: Given the sentence 'young caucasian hipster man holding tablet computer above the book and looking at butterflies flying out from the device,' extract meaningful triplets. Answer: \
    Meaningful triplets are <man, holding, computer>, <computer, above, book>, <man, looking at, butterflies>, and <butterflies, flying out, device>. \n\
    Question: Given the sentence 'black white and brown horses standing on the snow in a paddock near the white wooden fence,' extract meaningful triplets. Answer: \
    Meaningful triplets are <horses, standing on, snow>, <snow, in, paddock>, and <paddock, near, fence>. \n\
    Question: Given the sentence '{captions},' extract meaningful triplets. Answer: \
    "
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
    captions = caption_dict[int(i)]['caption']
    triplet_info[i] = {}
    triplet_info[i]['before_matched_triplets'] = []
    try:
        response, triplet = extract_triplets(captions)
        triplet_info[i]['before_matched_triplets'].extend(triplet)
    except:
        error_count += 1
        print("Error")
        continue

    if iter % save_period == 0:
        previous_path = f"dataset/CC/start_{start}_end_{end}_{iter-save_period}_original.pkl"
        if os.path.isfile(previous_path):
            os.remove(previous_path)
            
        with open(f"dataset/CC/start_{start}_end_{end}_{iter}_original.pkl", 'wb') as f:
            pickle.dump(triplet_info, f)

    if iter == end-start-1:
        previous_path = f"dataset/CC/start_{start}_end_{end}_{end-start-save_period}_original.pkl"
        if os.path.isfile(previous_path):
            os.remove(previous_path)
            
        with open(f"dataset/CC/start_{start}_end_{end}_final_original.pkl", 'wb') as f:
            pickle.dump(triplet_info, f)     
           

