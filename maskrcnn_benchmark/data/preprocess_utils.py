from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
#import collection_func

# Manually check & match the top-250 caption nouns
noun_manual_dict = {"group": "people", "couple": "people", "herd": "animal", "computer": "laptop",\
                    "dish": "plate", "suit": "jacket", "meat": "food", "monitor": "laptop", }
# Manually check & match the unmatched VG predicates as well as the top-100 caption predicates
rel_manual_dict = {"ride": "riding", "ride on": "riding", "sit on": "sitting on", "in front": "in front of", "underneath": "under", "play with": "playing with",\
                "attach": "attached to", "cover": "covered in",  "grow on": "growing on",\
                "fly": "flying in", "fly at": "flying in", "fly around": "flying in", "fly into": "flying in", "fly by": "flying in",\
                "hang over": "hanging from", "hang": "hanging from", "hang onto": "hanging from", "hang around": "hanging from",\
                "lay": "laying on",\
                "lie down": "lying on", "lie": "lying on", \
                "look into": "looking at", "look over": "looking at", "look in": "looking at", "look": "looking at", "look to": "looking at",\
                "mount": "mounted on", "back of": "on the back of", "paint": "painted on", "park": "parked on",\
                "stand by": "standing by", "stand around": "standing on", "stand": "standing on",\
                "walk on": "walking on", "walk into": "walking in", "walk": "walking in", "cross": "crossing"}

def wordnet_preprocess(vocab_A, vocab_B, pos=wn.NOUN, use_lemma=False):
    """
    Determine whether 2 given words are matched / have same meaning,
    by checking whether synsets have overlap, or whether it's hypernyms synset over the other.
    If use_lemma, plus checking whether the lemmas of the most-frequent synset match.
    """
    vocab_A = vocab_A.lower()
    vocab_B = vocab_B.lower()
    if pos == wn.NOUN:
        if len(vocab_A.split(" ")) != 1:
            vocab_A = vocab_A.split(" ")[-1]
        if len(vocab_B.split(" ")) != 1:
            vocab_B = vocab_B.split(" ")[-1]

    base_vocab_A = wn.morphy(vocab_A)
    base_vocab_B = wn.morphy(vocab_B)
    if base_vocab_A is None or base_vocab_B is None:
        return False

    if not use_lemma:
        synsets_A = wn.synsets(base_vocab_A, pos)
        synsets_B = wn.synsets(base_vocab_B, pos)

        # justify whether two synsets overlap with each other
        for s_a in synsets_A:
            for s_b in synsets_B:
                opt1 = s_a == s_b
                opt2 = len(list(set(s_a.lowest_common_hypernyms(s_b)).intersection(set([s_b])))) > 0
                s_a_lemma = [str(lemma.name()) for lemma in s_a.lemmas()]
                s_b_lemma = [str(lemma.name()) for lemma in s_b.lemmas()]
                overlap = [item for item in s_a_lemma if item in s_b_lemma]
                opt3 = len(overlap) > 0
                if opt1 or opt2 or opt3:
                    return True
        return False
    else:
        synsets_A = wn.synsets(base_vocab_A, pos)
        synsets_B = wn.synsets(base_vocab_B, pos)
        # synsets_A = [synsets_A[0]] if len(synsets_A) > 0 else [] # most frequent synset for given word
        synsets_B = [synsets_B[0]] if len(synsets_B) > 0 else [] # most frequent synset for given word

        # justify whether two synsets overlap with each other
        for s_a in synsets_A:
            for s_b in synsets_B:
                opt1 = s_a == s_b
                opt2 = len(list(set(s_a.lowest_common_hypernyms(s_b)).intersection(set([s_b])))) > 0
                s_a_lemma = [str(lemma.name()) for lemma in s_a.lemmas()]
                s_b_lemma = [str(lemma.name()) for lemma in s_b.lemmas()]
                overlap = [item for item in s_a_lemma if item in s_b_lemma]
                opt3 = len(overlap) > 0
                if opt1 or opt2 or opt3:
                    return True
        return False

def make_alias_dict(dict_file):
    """
    create an alias dictionary from a VG file: first word  in a line is the representative, the others belong to it.
    """
    out_dict = {}
    vocab = []
    for line in open(dict_file, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
        vocab.append(alias_target)
    return out_dict, vocab

def predicate_match(vocab_A, vocab_B, lem=None):
    """
    String matching of 2 predicte words.
    Use lemmatization on predicates since:
    (1) although the SG parser had returned the lemma of the predicates, there are still "holding" & "hold"
    (2) although the alias file will handle some of the predicate lemma, lemmatization can be additional chance
    (3) lemmatization has errors on predicate words, empirically
    """
    vocab_A = vocab_A.lower()
    vocab_B = vocab_B.lower()

    #The NLTK Lemmatization method is based on WordNet’s built-in morphy function.
    base_vocab_A = lem.lemmatize(vocab_A, 'v')
    base_vocab_B = lem.lemmatize(vocab_B, 'v')

    if base_vocab_A == base_vocab_B:
        return True
    else:
        return False

#####################################################################################################################
#### This file generate the Dict between VG concepts and caption concepts. The goal is to convert the predicted caption
#### categories into VG standard categories, for evaluation purpose. To this end, each caption concept can be only
#### matched to one of the VG categories, not vice versa. The rest caption concepts that didn't match to any of the
#### VG category will be ignored during ranking the triplets.
#### For nouns, the matching priority is as follows:
#### 1. Direct string match: caption "man" is converted into VG "man"
#### 2. Root string match: caption "baseball player" is converted into VG "player"
#### 3. Synset match: instead of lowest_common_hypernyms, the VG "room" must be hypernym of the caption "bathroom"
#### 0. Manual dict match: by looking at the results from the matching above, manually construct a dict for corner cases
#### For predicates, the matching priority is as follows:
#### 1. Direct string match: caption "on" is converted into VG "on"
#### 2. VG predicate alias file match: use the alias file from VG to match the predicates
#### 3. Lemmatization match: caption "have" is converted into VG "has"
#### 0. Manual dict match: by looking at the results from the matching above, manually construct a dict for corner cases
#####################################################################################################################

def object_word_map(word_list1, word_list2, method='synset'):
    use_lemma = True if method == "synset" else False # whether use lemma for synset matching
    index_mapped = [] # the index of word_list2 which has been mapped to some category in word_list1

    word_map = {} # the mapping of word (string), used for visualization
    cls_ind_map = {} # the mapping class index in dataset, used for model training and testing
    for i in tqdm(range(len(word_list1))):
        word_map[word_list1[i]] = []
        cls_ind_map[i] = []
        this_got_matched = False

        # Priority 0: manual dictionary match
        for j in range(len(word_list2)):
            if word_list1[i] in noun_manual_dict and noun_manual_dict[word_list1[i]] == word_list2[j]:
                word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                print('Manual Map: {} <-- {}'.format(word_list2[j], word_list1[i]))
                break

        # Priority 1: direct string match
        for j in range(len(word_list2)):
            if word_list1[i] == word_list2[j]:
                word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                break

        # Priority 2: root string match
        if not this_got_matched:
            for j in range(len(word_list2)):
                if word_list1[i].split()[-1] == word_list2[j].split()[-1]:
                    word_map[word_list1[i]].append(word_list2[j])
                    cls_ind_map[i].append(j)
                    index_mapped.append(j)
                    this_got_matched = True
                    break

        # Priority 3: synset match
        if not this_got_matched:
            for j in range(len(word_list2)):
                if method == "synset":
                    if wordnet_preprocess(word_list1[i], word_list2[j], pos=wn.NOUN, use_lemma=use_lemma):
                        word_map[word_list1[i]].append(word_list2[j])
                        cls_ind_map[i].append(j)
                        index_mapped.append(j)
                        print('Synset Map: {} <-- {}'.format(word_list2[j], word_list1[i]))

    # show the category in word_list2 which was matched successfully
    print("\nMatch successfully:")
    cnt = 0; unmatched_objs = []
    for i, item in enumerate(word_list2):
        if i in index_mapped:
            cnt += 1
        else:
            unmatched_objs.append(item)
    print(f"\nIn total, {cnt} of {len(word_list2)} categories were matched!\n")
    print(f"Unmatched objs: {unmatched_objs}")
    return word_map



def map_caption_concepts_to_gqa200_categories(parsed_objs, parsed_rels, vg_dataset):
    ############################################################################
    #### 1. map nouns
    ############################################################################
    method = "synset"
    word_list1 = parsed_objs  # root words
    word_list2 = vg_dataset.ind_to_classes  # all words to be merged
    print(f"Mapping {len(word_list1)} parsed nouns to {len(word_list2)} GQA150 obj categories...")

    word_map = object_word_map(word_list1, word_list2, method)

    ############################################################################
    #### 2. map predicates
    ############################################################################
    alias_file = "DATASET/VG150/relationship_alias.txt"
    lem = WordNetLemmatizer()
    index_mapped = [] # the index of word_list2 which has been mapped to some category in word_list1

    word_list1 = parsed_rels  # root words
    word_list2 = vg_dataset.ind_to_predicates  # all words to be merged
    print(f"Mapping {len(word_list1)} parsed rels to {len(word_list2)} VG150 rel categories...")

    # predicate alias dictionary
    alias_dict, vocab_list = make_alias_dict(alias_file)
    rel_word_map = {} # the mapping of word (string), used for visualization
    cls_ind_map = {} # the mapping class index in dataset, used for model training and testing
    for i in range(len(word_list1)):
        rel_word_map[word_list1[i]] = []
        cls_ind_map[i] = []
        this_got_matched = False

        # Priority 0: manual dictionary match
        for j in range(len(word_list2)):
            if word_list1[i] in rel_manual_dict and rel_manual_dict[word_list1[i]] == word_list2[j]:
                rel_word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                print('Manual Map: {} <-- {}'.format(word_list2[j], word_list1[i]))
                break

        # Priority 1: direct string match
        for j in range(len(word_list2)):
            if word_list1[i] == word_list2[j]:
                rel_word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                break

        # Priority 2: VG predicate alias file match
        if not this_got_matched:
            for j in range(len(word_list2)):
                if (word_list1[i] in alias_dict and alias_dict[word_list1[i]] == word_list2[j]) or\
                    (word_list2[j] in alias_dict and alias_dict[word_list2[j]] == word_list1[i]):
                    rel_word_map[word_list1[i]].append(word_list2[j])
                    cls_ind_map[i].append(j)
                    index_mapped.append(j)
                    this_got_matched = True
                    break

        # Priority 3: Lemmatization match
        if not this_got_matched:
            for j in range(len(word_list2)):
                if predicate_match(word_list1[i], word_list2[j], lem=lem):
                    rel_word_map[word_list1[i]].append(word_list2[j])
                    cls_ind_map[i].append(j)
                    index_mapped.append(j)
                    print('Lemmatization Map: {} <-- {}'.format(word_list2[j], word_list1[i]))
                    this_got_matched = True

    # show the category in word_list2 which was matched successfully
    print("\nMatch successfully:")
    cnt = 0; unmatched_rels = []
    for i, item in enumerate(word_list2):
        if i in index_mapped:
            cnt += 1
        else:
            unmatched_rels.append(item)
    print(f"\nIn total, {cnt} of {len(word_list2)} categories were matched!\n")
    print(f'Unmatched rels: {unmatched_rels}')

    return word_map, rel_word_map


def map_caption_concepts_to_vg150_categories(parsed_objs, parsed_rels, vg_dataset):
    ############################################################################
    #### 1. map nouns
    ############################################################################
    method = "synset"
    word_list1 = parsed_objs  # root words
    word_list2 = vg_dataset.ind_to_classes  # all words to be merged
    print(f"Mapping {len(word_list1)} parsed nouns to {len(word_list2)} VG150 obj categories...")

    word_map = object_word_map(word_list1, word_list2, method)

    ############################################################################
    #### 2. map predicates
    ############################################################################
    alias_file = "DATASET/VG150/relationship_alias.txt"
    lem = WordNetLemmatizer()
    index_mapped = [] # the index of word_list2 which has been mapped to some category in word_list1

    word_list1 = parsed_rels  # root words
    word_list2 = vg_dataset.ind_to_predicates  # all words to be merged
    print(f"Mapping {len(word_list1)} parsed rels to {len(word_list2)} VG150 rel categories...")

    # predicate alias dictionary
    alias_dict, vocab_list = make_alias_dict(alias_file)
    rel_word_map = {} # the mapping of word (string), used for visualization
    cls_ind_map = {} # the mapping class index in dataset, used for model training and testing
    for i in range(len(word_list1)):
        rel_word_map[word_list1[i]] = []
        cls_ind_map[i] = []
        this_got_matched = False

        # Priority 0: manual dictionary match
        for j in range(len(word_list2)):
            if word_list1[i] in rel_manual_dict and rel_manual_dict[word_list1[i]] == word_list2[j]:
                rel_word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                print('Manual Map: {} <-- {}'.format(word_list2[j], word_list1[i]))
                break

        # Priority 1: direct string match
        for j in range(len(word_list2)):
            if word_list1[i] == word_list2[j]:
                rel_word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                break

        # Priority 2: VG predicate alias file match
        if not this_got_matched:
            for j in range(len(word_list2)):
                if (word_list1[i] in alias_dict and alias_dict[word_list1[i]] == word_list2[j]) or\
                    (word_list2[j] in alias_dict and alias_dict[word_list2[j]] == word_list1[i]):
                    rel_word_map[word_list1[i]].append(word_list2[j])
                    cls_ind_map[i].append(j)
                    index_mapped.append(j)
                    this_got_matched = True
                    break

        # Priority 3: Lemmatization match
        if not this_got_matched:
            for j in range(len(word_list2)):
                if predicate_match(word_list1[i], word_list2[j], lem=lem):
                    rel_word_map[word_list1[i]].append(word_list2[j])
                    cls_ind_map[i].append(j)
                    index_mapped.append(j)
                    print('Lemmatization Map: {} <-- {}'.format(word_list2[j], word_list1[i]))
                    this_got_matched = True

    # show the category in word_list2 which was matched successfully
    print("\nMatch successfully:")
    cnt = 0; unmatched_rels = []
    for i, item in enumerate(word_list2):
        if i in index_mapped:
            cnt += 1
        else:
            unmatched_rels.append(item)
    print(f"\nIn total, {cnt} of {len(word_list2)} categories were matched!\n")
    print(f'Unmatched rels: {unmatched_rels}')

    return word_map, rel_word_map


def map_caption_concepts_to_vg150_categories_using_LLM(parsed_objs, parsed_rels, vg_dataset):
    ############################################################################
    #### 1. map nouns - LLM Version
    ############################################################################
    word_list1 = parsed_objs  # root words
    entity_set = vg_dataset.ind_to_classes  # all words to be merged
    print(f"Mapping {len(word_list1)} parsed nouns to {len(entity_set)} VG150 obj categories...")

    
    index_mapped = [] # the index of word_list2 which has been mapped to some category in word_list1

    word_map = {} # the mapping of word (string), used for visualization
    cls_ind_map = {} # the mapping class index in dataset, used for model training and testing
    for i in tqdm(len(word_list1)):
        this_got_matched = False
        word_map[word_list1[i]] = []
        cls_ind_map[i] = []
        
        # Direct Mathching
        if word_list1[i] in entity_set:
            word_map[word_list1[i]].append(word_list1[i])
            index_mapped.append(entity_set.index(word_list1[i]))
            cls_ind_map[i].append(entity_set.index(word_list1[i]))
            break
        
        # Manual Directory Matching
        for j in range(len(entity_set)):
            if word_list1[i] in noun_manual_dict and noun_manual_dict[word_list1[i]] == entity_set[j]:
                word_map[word_list1[i]].append(entity_set[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                print('Manual Map: {} <-- {}'.format(entity_set[j], word_list1[i]))
                break

        if not this_got_matched:
            ent = collection_func.match_entity(word_list1[i])
            if ent in entity_set:
                word_map[word_list1[i]].append(ent)
                index_mapped.append(entity_set.index(ent))
                cls_ind_map[i].append(entity_set.index(ent))
                
    cnt_obj = 0; unmatched_objs = []
    for i, item in enumerate(entity_set):
        if i in index_mapped:
            cnt_obj+=1
        else:
            unmatched_objs.append(item)
    ############################################################################
    #### 2. map predicates - LLM Version
    ############################################################################
    index_mapped = [] # the index of word_list2 which has been mapped to some category in word_list1

    word_list1 = parsed_rels  # root words
    predicate_set = vg_dataset.ind_to_predicates  # all words to be merged
    print(f"Mapping {len(word_list1)} parsed rels to {len(predicate_set)} VG150 rel categories...")

    rel_word_map = {} # the mapping of word (string), used for visualization
    cls_ind_map = {} # the mapping class index in dataset, used for model training and testing
    for i in range(len(word_list1)):
        rel_word_map[word_list1[i]] = []
        cls_ind_map[i] = []
        this_got_matched = False
        
        # Direct Matching
        if word_list1[i] in predicate_set:
            rel_word_map[word_list1[i]].append(predicate_set[j])
            cls_ind_map[i].append(j)
            index_mapped.append(j)
            break
        
        # Priority 0: manual dictionary match
        for j in range(len(predicate_set)):
            if word_list1[i] in rel_manual_dict and rel_manual_dict[word_list1[i]] == predicate_set[j]:
                rel_word_map[word_list1[i]].append(predicate_set[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                print('Manual Map: {} <-- {}'.format(predicate_set[j], word_list1[i]))
                break
            
        # LLM
        if not this_got_matched:
            pred = collection_func.match_relation(word_list1[i])
            if pred in predicate_set:
                rel_word_map[word_list1[i]].append(pred)
                index_mapped.append(predicate_set.index(pred))
                cls_ind_map[i].append(predicate_set.index(pred))
                

    # show the category in word_list2 which was matched successfully
    print("\nMatch successfully:")
    cnt = 0; unmatched_rels = []
    for i, item in enumerate(predicate_set):
        if i in index_mapped:
            cnt += 1
        else:
            unmatched_rels.append(item)
    print(f"\nIn total, {cnt} of {len(predicate_set)} categories were matched!\n")
    print(f'Unmatched rels: {unmatched_rels}')

    return word_map, rel_word_map



if __name__ == '__main__':
    VG150_OBJ_CATEGORIES = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
    VG150_BASE_OBJ_CATEGORIES = ['__background__', 'tile', 'drawer', 'men', 'railing', 'stand', 'towel', 'sneaker', 'vegetable', 'screen', 'vehicle', 'animal', 'kite', 'cabinet', 'sink', 'wire', 'fruit', 'curtain', 'lamp', 'flag', 'pot', 'sock', 'boot', 'guy', 'kid', 'finger', 'basket', 'wave', 'lady', 'orange', 'number', 'toilet', 'post', 'room', 'paper', 'mountain', 'paw', 'banana', 'rock', 'cup', 'hill', 'house', 'airplane', 'plant', 'skier', 'fork', 'box', 'seat', 'engine', 'mouth', 'letter', 'windshield', 'desk', 'board', 'counter', 'branch', 'coat', 'logo', 'book', 'roof', 'tie', 'tower', 'glove', 'sheep', 'neck', 'shelf', 'bottle', 'cap', 'vase', 'racket', 'ski', 'phone', 'handle', 'boat', 'tire', 'flower', 'child', 'bowl', 'pillow', 'player', 'trunk', 'bag', 'wing', 'light', 'laptop', 'pizza', 'cow', 'truck', 'jean', 'eye', 'arm', 'leaf', 'bird', 'surfboard', 'umbrella', 'food', 'people', 'nose', 'beach', 'sidewalk', 'helmet', 'face', 'skateboard', 'motorcycle', 'clock', 'bear']
    while True:
        wordnet_preprocess('woman', 'man', use_lemma=False)
        orglabel2base = object_word_map(VG150_OBJ_CATEGORIES, VG150_BASE_OBJ_CATEGORIES)
        # print([k for k, v in orglabel2base.items() if len(v) == 0])
        # print({k: v for k, v in orglabel2base.items() if k not in VG150_BASE_OBJ_CATEGORIES})

