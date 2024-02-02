import json
import logging
import os
import pickle
import random
from collections import defaultdict, OrderedDict, Counter

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList
from .od_to_grounding import convert_od_to_grounding_simple, convert_od_to_grounding_with_pseudo_triplet_caption, convert_od_to_grounding_simple_pred
from .modulated_coco import create_positive_map, create_positive_map_for_od_labels, create_greenlight_map


HEAD = []
BODY = []
TAIL = []

for i, cate in enumerate(cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT):
    if cate == 'h':
        HEAD.append(i)
    elif cate == 'b':
        BODY.append(i)
    elif cate == 't':
        TAIL.append(i)


def load_cate_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    ind_to_predicates_cate = ['__background__'] + info['rel']
    ind_to_entites_cate = ['__background__'] + info['obj']

    # print(len(ind_to_predicates_cate))
    # print(len(ind_to_entites_cate))
    predicate_to_ind = {idx: name for idx, name in enumerate(ind_to_predicates_cate)}
    entites_cate_to_ind = {idx: name for idx, name in enumerate(ind_to_entites_cate)}

    return (ind_to_entites_cate, ind_to_predicates_cate,
            entites_cate_to_ind, predicate_to_ind)


def load_annotations(annotation_file, img_dir, num_img, split,
                    filter_empty_rels, ):
    """

    :param annotation_file:
    :param img_dir:
    :param img_range:
    :param filter_empty_rels:
    :return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """

    annotations = json.load(open(annotation_file, 'r'))

    if num_img == -1 :
        num_img = len(annotations)

    annotations = annotations[: num_img ]

    empty_list = set()
    if filter_empty_rels:
        for i, each in enumerate(annotations):
            if len(each['rel']) == 0:
                empty_list.add(i)
            if len(each['bbox']) == 0:
                empty_list.add(i)

    print('empty relationship image num: ', len(empty_list))


    boxes = []
    gt_classes = []
    relationships = []
    img_info = []
    for i, anno in enumerate(annotations):

        if i in empty_list:
            continue

        boxes_i = np.array(anno['bbox'])
        gt_classes_i = np.array(anno['det_labels'], dtype=int)

        rels = np.array(anno['rel'], dtype=int)

        gt_classes_i += 1
        rels[:, -1] += 1

        image_info = {
            'width': anno['img_size'][0],
            'height': anno['img_size'][1],
            'img_fn': os.path.join(img_dir, anno['img_fn'] + '.jpg')
        }

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)
        img_info.append(image_info)


    return boxes, gt_classes, relationships, img_info



class OIDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, ann_file, cate_info_file, tokenizer = None, transforms=None,
                 num_im=-1, check_img_file=False, filter_duplicate_rels=True,  flip_aug=False):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        self.cfg=cfg

        #
        # num_im = 20000
        # num_val_im = 1000

        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split
        # cfg.DATA_DIR = '/home/public/Datasets/CV/open-imagev6'
        img_dir = cfg.DATA_DIR
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.cate_info_file =cfg.DATA_DIR +'/annotations/categories_dict.json'
        self.annotation_file = cfg.DATA_DIR + '/annotations/' +ann_file.split('/')[-1]
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.repeat_dict = None
        self.check_img_file = check_img_file
        self.remove_tail_classes = False
        self.vg_cat_dict = json.load(open(cfg.DATA_DIR + '/annotations/OI-SGG-Category.json', 'r'))
        (self.ind_to_classes,
         self.ind_to_predicates,
         self.classes_to_ind,
         self.predicates_to_ind) = load_cate_info(self.cate_info_file)  # contiguous 151, 51 containing __background__

        logger = logging.getLogger("pysgg.dataset")
        self.logger = logger

        #self.categories = {i: self.ind_to_classes[i]
        #                   for i in range(len(self.ind_to_classes))}

        self.gt_boxes, self.gt_classes, self.relationships, self.img_info,= load_annotations(
            self.annotation_file, img_dir + '/images', num_im, split=split,
            filter_empty_rels=False if not cfg.MODEL.RELATION_ON and split == "train" else True,
        )
        self.filenames = [img_if['img_fn'] for img_if in self.img_info]
        self.idx_list = list(range(len(self.filenames)))

        self.id_to_img_map = {k: v for k, v in enumerate(self.idx_list)}

    def categories(self):
        cats_no_bg = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes)) if i > 0} # all categories
        return cats_no_bg



    def __getitem__(self, index):

        img = Image.open(self.filenames[index]).convert("RGB")
        org_img_size = img.size
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)
        target = self.get_groundtruth(index)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
          
        target = self._add_pseudo_caption(index, target, org_img_size)
        target.add_field('filename', self.filenames[index])
        
        return img, target, index

    def _add_pseudo_caption(self, index, target, org_img_size, max_query_len=1210):

        annotations, caption, greenlight_span_for_masked_lm_objective = convert_od_to_grounding_simple(
            target=target,
            image_id=index,
            ind_to_class={i: n for i, n in enumerate(self.ind_to_classes) if i > 0},
            separation_tokens='. ',
        )

        # image info
        target.add_field("image_id", index)
        target.add_field("area", torch.tensor([obj["area"] for obj in annotations]))
        target.add_field("iscrowd", torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations]))
        w, h = org_img_size
        target.add_field("orig_size", torch.as_tensor([int(h), int(w)]))
        target.add_field("size", torch.as_tensor([int(h), int(w)]))

        # pseudo caption
        tokens_positive = [obj["tokens_positive"] for obj in annotations]
        target.add_field("caption", caption)
        target.add_field("tokens_positive", tokens_positive)
        tokenized = self.tokenizer(caption, return_tensors="pt", max_length=max_query_len, truncation=True)
        target.add_field("positive_map", create_positive_map(tokenized, tokens_positive, max_len=max_query_len))
        target.add_field('greenlight_map', create_greenlight_map(greenlight_span_for_masked_lm_objective, tokenized, max_len=max_query_len))
        target.add_field("positive_map_for_od_labels", create_positive_map_for_od_labels(tokenized, {}, max_len=max_query_len))

        return target



    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False):

        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        box = self.gt_boxes[index]
        box = torch.from_numpy(box)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:, 2]
            new_xmax = w - box[:, 0]
            box[:, 0] = new_xmin
            box[:, 2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(np.zeros((len(self.gt_classes[index]), 10) ) ))


        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v))
                        for k, v in all_rel_sets.items()]
 
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i,0]), int(relation[i,1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
            else:
                relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
        target.add_field("relation", relation_map)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation)) # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        return len(self.idx_list)
