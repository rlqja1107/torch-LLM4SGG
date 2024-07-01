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
    From the given sentence, the task is to extract meaningful triplets formed as <subject, predicate, object>.\
    To extract meaningful triplets from the sentence, please follow the following two steps. \
    Step 1: Paraphrase the sentence. \n\
    Step 2: From the paraphrased sentence obtained in the Step 1, extract meaningful triplets formed as <subject, predicate, object>. Note that the subject is the entity or noun that performs the action or is being described, and the object is the entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being). \n\
    Let's take a few examples to understand how to extract meaningful triplets. \
    Question: Given the sentence 'person walking in cherry red boots and carrying a large see through shopping bag,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'An individual strolls while wearing bright cherry red boots and carries a large transparent shopping bag'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence are: <individual, wearing, boot>, <individual, carrying, bag>. \
    Output: <individual, wearing, boot>, <individual, carrying, bag>. \n\
    Question: Given the sentence 'a headless mannequin in a leather jacket with a sign attached to it,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'A headless mannequin wearing a leather jacket has a sign affixed to it'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence are: <mannequin, wearing, jacket>, <mannequin, has, sign>, <sign, mounted on, it>. \
    Output: <mannequin, wearing, jacket>, <mannequin, has, sign>, <sign, mounted on, it>. \n\
    Question: Given the sentence 'Four clock sitting on a floor next to a woman's feet,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'Four clocks are placed on the floor beside a woman's feet'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <clock, placed on, floor>, <clock, beside, feet>. \
    Output: <clock, placed on, floor>, <clock, beside, feet>. \n\
    Question: Given the sentence 'baby boy lying on his front on a bed raising his head and shoulders up and looking at the camera,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'A baby boy lies on his stomach on a bed, lifting his head and shoulders while gazing at the camera'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <boy, lying on, stmoach>, <stomach, on, bed>, <boy, lifting, head>, <boy, lifting, shoulder>, <boy, gazing at, camera>. \
    Output: <boy, lying on, stmoach>, <stomach, on, bed>, <boy, lifting, head>, <boy, lifting, shoulder>, <boy, gazing at, camera>. \n\
    Question: Given the sentence 'a man dressed in a red suit covered in trees, with a tie to match,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'A man wearing a red suit adorned with trees, complete with a matching tie'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <man, wearing, suit>, <suit, adorned with, tree>, <man, with, tie>. \
    Output: <man, wearing, suit>, <suit, adorned with, tree>, <man, with, tie>. \n\
    Question: Given the sentence 'Individuals stroll along the shoreline with the waves beneath their feet,' extract meaningful triplets. Answer: \
    Step 1: The sentence can be paraphrased as: 'people walking along the beach with the waves beneath their feet'. \
    Step 2: Meaningful triplets, where the subject and object are the simple noun, extracted from the paraphrased sentence obtained in the Step 1 are: <people, walking along, beach>, <beach, with, wave>, <wave, beneath, feet>. \
    Output: <people, walking along, beach>, <beach, with, wave>, <wave, beneath, feet>. \n\
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
    caption = caption_dict[int(i)]['caption']
    try:
        response, (triplet, paraphrased_cap) = extract_triplets(caption)
        triplet_info[i] = {}
        triplet_info[i]['p_sentence'] = []
        triplet_info[i]['before_matched_triplets'] = []
        
        triplet_info[i]['before_matched_triplets'].extend(triplet)
        triplet_info[i]['p_sentence'].extend(paraphrased_cap)
    except:
        error_count += 1
        print("Error")
        continue
        
    if iter % save_period == 0:
        previous_path = f"dataset/CC/start_{start}_end_{end}_{iter-save_period}_paraphrase.pkl"
        if os.path.isfile(previous_path):
            os.remove(previous_path)
            
        with open(f"dataset/CC/start_{start}_end_{end}_{iter}_paraphrase.pkl", 'wb') as f:
            pickle.dump(triplet_info, f)
            
  
    if iter == end-start-1:
        previous_path = f"dataset/CC/start_{start}_end_{end}_{end-start-save_period}_paraphrase.pkl"
        if os.path.isfile(previous_path):
            os.remove(previous_path)
            
        with open(f"dataset/CC/start_{start}_end_{end}_final_paraphrase.pkl", 'wb') as f:
            pickle.dump(triplet_info, f)     