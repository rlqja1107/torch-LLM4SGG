import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import reduce

import numpy as np
import torch
from sklearn import metrics

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation.vg.vg_stage_eval_utils import (
    boxlist_iou,
    intersect_2d_torch_tensor,
    dump_hit_indx_dict_to_tensor,
    trans_cluster_label,
    ENTITY_CLUSTER,
    PREDICATE_CLUSTER,
)
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps


class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict

    @abstractmethod
    def register_container(self, mode):
        print("Register Result Container")
        pass

    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass


"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""


def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns:
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack(
            (
                class_scores[sub_id],
                predicate_scores,
                class_scores[ob_id],
            )
        )

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(
    gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thres, phrdet=False
):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(
        np.where(gt_has_match)[0],
        gt_boxes[gt_has_match],
        keeps[gt_has_match],
    ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate(
                (gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0
            )

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


class SGRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGRecall, self).__init__(result_dict)
        self.type = "recall"

    def register_container(self, mode):
        self.result_dict[mode + f"_{self.type}"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_recall"].items():
            result_str += "  R @ %d: %.4f; " % (k, np.mean(v))
        result_str += " for mode=%s, type=Recall(Main)." % mode
        result_str += "\n"
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container["pred_rel_inds"]
        rel_scores = local_container["rel_scores"]
        gt_rels = local_container["gt_rels"]
        gt_classes = local_container["gt_classes"]
        gt_boxes = local_container["gt_boxes"]
        pred_classes = local_container["pred_classes"]
        pred_boxes = local_container["pred_boxes"]
        obj_scores = local_container["obj_scores"]

        iou_thres = global_container["iou_thres"]

        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        pred_scores = rel_scores[:, 1:].max(1)

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container["gt_triplets"] = gt_triplets
        local_container["gt_triplet_boxes"] = gt_triplet_boxes

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
            pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores
        )

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode == "phrdet",
        )
        local_container["pred_to_gt"] = pred_to_gt

        for k in self.result_dict[mode + "_recall"]:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + "_recall"][k].append(rec_i)

        return local_container


"""
No Graph Constraint Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""


class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGNoGraphConstraintRecall, self).__init__(result_dict)
        self.type = "recall_nogc"

    def register_container(self, mode):
        self.result_dict[mode + "_recall_nogc"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_recall_nogc"].items():
            result_str += "ngR @ %d: %.4f; " % (k, np.mean(v))
        result_str += " for mode=%s, type=No Graph Constraint Recall(Main)." % mode
        result_str += "\n"
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container["obj_scores"]
        pred_rel_inds = local_container["pred_rel_inds"]
        rel_scores = local_container["rel_scores"]
        pred_boxes = local_container["pred_boxes"]
        pred_classes = local_container["pred_classes"]
        gt_rels = local_container["gt_rels"]

        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack(
            (pred_rel_inds[nogc_score_inds[:, 0]], nogc_score_inds[:, 1] + 1)
        )
        nogc_pred_scores = rel_scores[nogc_score_inds[:, 0], nogc_score_inds[:, 1] + 1]

        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(
            nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores
        )

        # No Graph Constraint
        gt_triplets = local_container["gt_triplets"]
        gt_triplet_boxes = local_container["gt_triplet_boxes"]
        iou_thres = global_container["iou_thres"]

        nogc_pred_to_gt = _compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            phrdet=mode == "phrdet",
        )

        local_container["nogc_pred_to_gt"] = nogc_pred_to_gt

        for k in self.result_dict[mode + "_recall_nogc"]:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + "_recall_nogc"][k].append(rec_i)


"""
Zero Shot Scene Graph
Only calculate the triplet that not occurred in the training set
"""


class SGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGZeroShotRecall, self).__init__(result_dict)
        self.type = "zeroshot_recall"

    def register_container(self, mode):
        self.result_dict[mode + "_zeroshot_recall"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_zeroshot_recall"].items():
            result_str += " zR @ %d: %.4f; " % (k, np.mean(v))
        result_str += " for mode=%s, type=Zero Shot Recall." % mode
        result_str += "\n"
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container["gt_rels"]
        gt_classes = local_container["gt_classes"]
        zeroshot_triplets = global_container["zeroshot_triplet"]

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack(
            (gt_classes[sub_id], gt_classes[ob_id], pred_label)
        )  # num_rel, 3

        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[
            0
        ].tolist()

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container["pred_to_gt"]

        for k in self.result_dict[mode + "_zeroshot_recall"]:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = (
                    len(self.zeroshot_idx)
                    + len(match_list)
                    - len(set(self.zeroshot_idx + match_list))
                )
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + "_zeroshot_recall"][k].append(zero_rec_i)


"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""


class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGPairAccuracy, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + "_accuracy_hit"] = {20: [], 50: [], 100: []}
        self.result_dict[mode + "_accuracy_count"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_accuracy_hit"].items():
            a_hit = np.mean(v)
            a_count = np.mean(self.result_dict[mode + "_accuracy_count"][k])
            result_str += "  A @ %d: %.4f; " % (k, a_hit / a_count)
        result_str += " for mode=%s, type=TopK Accuracy." % mode
        result_str += "\n"
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = (
            local_container["pred_rel_inds"][:, 0] * 1024
            + local_container["pred_rel_inds"][:, 1]
        )
        gt_pair_idx = (
            local_container["gt_rels"][:, 0] * 1024 + local_container["gt_rels"][:, 1]
        )
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container["pred_to_gt"]
        gt_rels = local_container["gt_rels"]

        for k in self.result_dict[mode + "_accuracy_hit"]:
            # to calculate accuracy, only consider those gt pairs
            # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
            # for sgcls and predcls
            if mode != "sgdet":
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[mode + "_accuracy_hit"][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + "_accuracy_count"][k].append(float(gt_rels.shape[0]))


"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""


class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]  # remove __background__
        self.type = "mean_recall"

    def register_container(self, mode):
        # self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        # self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + "_mean_recall"] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + "_mean_recall_collect"] = {
            20: [[] for i in range(self.num_rel)],
            50: [[] for i in range(self.num_rel)],
            100: [[] for i in range(self.num_rel)],
        }
        self.result_dict[mode + "_mean_recall_list"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_mean_recall"].items():
            result_str += " mR @ %d: %.4f; " % (k, float(v))
        result_str += " for mode=%s, type=Mean Recall." % mode
        result_str += "\n"
        if self.print_detail:
            result_str += "Per-class recall@20: \n"
            for n, r in zip(
                self.rel_name_list, self.result_dict[mode + "_mean_recall_list"][20]
            ):
                result_str += "({}:{:.4f}) ".format(str(n), r)
            result_str += "\n"
            result_str += "Per-class recall@50: \n"
            for n, r in zip(
                self.rel_name_list, self.result_dict[mode + "_mean_recall_list"][50]
            ):
                result_str += "({}:{:.4f}) ".format(str(n), r)
            result_str += "\n"
            result_str += "Per-class recall@100: \n"
            for n, r in zip(
                self.rel_name_list, self.result_dict[mode + "_mean_recall_list"][100]
            ):
                result_str += "({}:{:.4f}) ".format(str(n), r)
            result_str += "\n\n"

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container["pred_to_gt"]
        gt_rels = local_container["gt_rels"]

        for k in self.result_dict[mode + "_mean_recall_collect"]:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1

            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + "_mean_recall_collect"][k][n].append(
                        float(recall_hit[n] / recall_count[n])
                    )

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + "_mean_recall"].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + "_mean_recall_collect"][k][idx + 1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(
                        self.result_dict[mode + "_mean_recall_collect"][k][idx + 1]
                    )
                self.result_dict[mode + "_mean_recall_list"][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + "_mean_recall"][k] = sum_recall / float(num_rel_no_bg)
        return


class SGNGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGNGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]  # remove __background__
        self.type = "ng_mean_recall"

    def register_container(self, mode):
        self.result_dict[mode + "_ng_mean_recall"] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + "_ng_mean_recall_collect"] = {
            20: [[] for i in range(self.num_rel)],
            50: [[] for i in range(self.num_rel)],
            100: [[] for i in range(self.num_rel)],
        }
        self.result_dict[mode + "_ng_mean_recall_list"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_ng_mean_recall"].items():
            result_str += "ng-mR @ %d: %.4f; " % (k, float(v))
        result_str += " for mode=%s, type=No Graph Constraint Mean Recall." % mode
        result_str += "\n"
        if self.print_detail:
            result_str += "----------------------- Details ------------------------\n"
            for n, r in zip(
                self.rel_name_list, self.result_dict[mode + "_ng_mean_recall_list"][100]
            ):
                result_str += "({}:{:.4f}) ".format(str(n), r)
            result_str += "\n"
            result_str += "--------------------------------------------------------\n"

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container["nogc_pred_to_gt"]
        gt_rels = local_container["gt_rels"]

        for k in self.result_dict[mode + "_ng_mean_recall_collect"]:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1

            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + "_ng_mean_recall_collect"][k][n].append(
                        float(recall_hit[n] / recall_count[n])
                    )

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + "_ng_mean_recall"].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + "_ng_mean_recall_collect"][k][idx + 1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(
                        self.result_dict[mode + "_ng_mean_recall_collect"][k][idx + 1]
                    )
                self.result_dict[mode + "_ng_mean_recall_list"][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + "_ng_mean_recall"][k] = sum_recall / float(num_rel_no_bg)
        return


"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""


class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + "_accumulate_recall"] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_accumulate_recall"].items():
            result_str += " aR @ %d: %.4f; " % (k, float(v))
        result_str += " for mode=%s, type=Accumulate Recall." % mode
        result_str += "\n"
        return result_str

    def calculate_accumulate(self, mode):
        for k, v in self.result_dict[mode + "_accumulate_recall"].items():
            self.result_dict[mode + "_accumulate_recall"][k] = float(
                self.result_dict[mode + "_recall_hit"][k][0]
            ) / float(self.result_dict[mode + "_recall_count"][k][0] + 1e-10)

        return


class SGStagewiseRecall(SceneGraphEvaluation):
    def __init__(
        self,
        result_dict,
    ):
        super(SGStagewiseRecall, self).__init__(result_dict)
        self.type = "stage_recall"

        # the recall statistic for each categories
        # for the following visualization
        self.per_img_rel_cls_recall = []
        for _ in range(3):
            self.per_img_rel_cls_recall.append(
                {
                    "pair_loc": [],
                    "pair_det": [],
                    "rel_hit": [],
                    "pred_cls": [],
                }
            )

        self.relation_per_cls_hit_recall = {
            "rel_hit": torch.zeros(
                (3, cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES, 2), dtype=torch.int64
            ),
            "pair_loc": torch.zeros(
                (3, cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES, 2), dtype=torch.int64
            ),
            "pair_det": torch.zeros(
                (3, cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES, 2), dtype=torch.int64
            ),
            "pred_cls": torch.zeros(
                (3, cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES, 2), dtype=torch.int64
            ),
        }

        self.rel_hit_types = [
            "pair_loc",
            "pair_det",
            "pred_cls",
            "rel_hit",
        ]
        self.eval_rel_pair_prop = True
        if cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PAIR_NUMS_AFTER_FILTERING < 0:
            self.eval_rel_pair_prop = cfg.MODEL.ROI_RELATION_HEAD.MAX_PROPOSAL_PAIR

        self.rel_pn_on = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON

        self.vaild_rel_prop_num = 300
        if cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.MP_ON_VALID_PAIRS:
            self.vaild_rel_prop_num = (
                cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.MP_VALID_PAIRS_NUM
            )

        self.mp_pair_refine_iter = 1
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "BGNNPredictor":
            self.mp_pair_refine_iter = (
                cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.ITERATE_MP_PAIR_REFINE
            )

        elif cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "GPSNetPredictor":
            self.mp_pair_refine_iter = (
                cfg.MODEL.ROI_RELATION_HEAD.GPSNET_MODULE.ITERATE_MP_PAIR_REFINE
            )


        # todo category clustering for overlapping
        self.instance_class_clustering = False
        self.predicate_class_clustering = False

    def register_container(self, mode):
        # the recall value for each images

        self.result_dict[f"{mode}_{self.type}_pair_loc"] = {20: [], 50: [], 100: []}
        self.result_dict[f"{mode}_{self.type}_pair_det"] = {20: [], 50: [], 100: []}
        self.result_dict[f"{mode}_{self.type}_rel_hit"] = {20: [], 50: [], 100: []}
        self.result_dict[f"{mode}_{self.type}_pred_cls"] = {20: [], 50: [], 100: []}
        self.result_dict[f"{mode}_{self.type}_rel_prop_pair_loc_before_relrpn"] = []
        self.result_dict[f"{mode}_{self.type}_rel_prop_pair_det_before_relrpn"] = []
        self.result_dict[f"{mode}_{self.type}_rel_prop_pair_loc_after_relrpn"] = []
        self.result_dict[f"{mode}_{self.type}_rel_prop_pair_det_after_relrpn"] = []

        for i in range(self.mp_pair_refine_iter):
            self.result_dict[
                f"{mode}_{self.type}_rel_pn_ap-iter{i}-top{self.vaild_rel_prop_num}"
            ] = []
            self.result_dict[f"{mode}_{self.type}_rel_pn_ap-iter{i}-top100"] = []

            self.result_dict[
                f"{mode}_{self.type}_rel_pn_auc-iter{i}-top{self.vaild_rel_prop_num}"
            ] = []
            self.result_dict[f"{mode}_{self.type}_rel_pn_auc-iter{i}-top100"] = []

        self.result_dict[f"{mode}_{self.type}_pred_cls_auc-top100"] = []
        self.result_dict[f"{mode}_{self.type}_effective_union_pairs_rate"] = []
        self.result_dict[f"{mode}_{self.type}_effective_union_pairs_range"] = []
        self.result_dict[f"{mode}_instances_det_recall"] = []
        self.result_dict[f"{mode}_instances_loc_recall"] = []

        # todo add per cls evaluation

    def generate_print_string(self, mode):
        result_str = "SGG Stagewise Recall: \n"
        for each_rel_hit_type in self.rel_hit_types:
            result_str += "    "
            if isinstance(self.result_dict[f"{mode}_{self.type}_{each_rel_hit_type}"], dict):
                iter_obj = self.result_dict[f"{mode}_{self.type}_{each_rel_hit_type}"].items()
            else:
                iter_obj = [
                    (cfg.MODEL.ROI_RELATION_HEAD.MAX_PROPOSAL_PAIR, vals)
                    for vals in self.result_dict[f"{mode}_{self.type}_{each_rel_hit_type}"]
                ]
            for k, v in iter_obj:
                result_str += " R @ %d: %.4f; " % (k, float(np.mean(v)))
            result_str += f" for mode={mode}, type={each_rel_hit_type}"
            result_str += "\n"
        result_str += "\n"

        result_str += (
            "instances detection recall:\n"
            f"locating: {np.mean(self.result_dict[f'{mode}_instances_loc_recall']):.4f}\n"
            f"detection: {np.mean(self.result_dict[f'{mode}_instances_det_recall']):.4f}\n"
        )
        result_str += "\n"

        if self.eval_rel_pair_prop:
            result_str += "effective relationship union pairs statistics \n"
            result_str += (
                f"effective relationship union pairs_rate (avg): "
                f"{np.mean(self.result_dict[f'{mode}_{self.type}_effective_union_pairs_rate']) : .3f}\n"
            )

            result_str += (
                f"effective relationship union pairs range(avg(percentile_85)/total): "
                f"{int(np.mean(self.result_dict[f'{mode}_{self.type}_effective_union_pairs_range']) + 1)}"
                f"({int(np.percentile(self.result_dict[f'{mode}_{self.type}_effective_union_pairs_range'], 85))}) / "
                f"{self.eval_rel_pair_prop} \n\n"
            )

            for i in range(self.mp_pair_refine_iter):
                if len(self.result_dict[f"{mode}_{self.type}_rel_pn_auc-iter{i}-top100"]) > 0:
                    result_str += (
                        f"The AUC of relpn (stage {i})-top100: "
                        f"{np.mean(self.result_dict[f'{mode}_{self.type}_rel_pn_auc-iter{i}-top100']): .3f} \n"
                    )

                if len(self.result_dict[f"{mode}_{self.type}_rel_pn_ap-iter{i}-top100"]) > 0:
                    result_str += (
                        f"The AP of relpn (stage {i})-top100: "
                        f"{np.mean(self.result_dict[f'{mode}_{self.type}_rel_pn_ap-iter{i}-top100']): .3f} \n"
                    )

                if (
                    len(
                        self.result_dict[
                            f"{mode}_{self.type}_rel_pn_auc-iter{i}-top{self.vaild_rel_prop_num}"
                        ]
                    )
                    > 0
                ):
                    result_str += (
                        f"The AUC of relpn (stage {i})-top{self.vaild_rel_prop_num}: "
                        f"{np.mean(self.result_dict[f'{mode}_{self.type}_rel_pn_auc-iter{i}-top{self.vaild_rel_prop_num}']): .3f} \n"
                    )

                if (
                    len(
                        self.result_dict[
                            f"{mode}_{self.type}_rel_pn_ap-iter{i}-top{self.vaild_rel_prop_num}"
                        ]
                    )
                    > 0
                ):
                    result_str += (
                        f"The AP of relpn (stage {i})-top{self.vaild_rel_prop_num}: "
                        f"{np.mean(self.result_dict[f'{mode}_{self.type}_rel_pn_ap-iter{i}-top{self.vaild_rel_prop_num}']): .3f} \n"
                    )

        if len(self.result_dict[f"{mode}_{self.type}_pred_cls_auc-top100"]) > 0:
            result_str += (
                f"The AUC of pred_clssifier: "
                f"{np.mean(self.result_dict[f'{mode}_{self.type}_pred_cls_auc-top100']): .3f} \n"
            )

        result_str += "\n"

        return result_str

    def calculate_recall(
        self,
        mode,
        global_container,
        gt_boxlist,
        gt_relations,
        pred_boxlist,
        pred_rel_pair_idx,
        pred_rel_scores,
    ):
        """
        evaluate stage-wise recall on one images

        :param global_container:
        :param gt_boxlist: ground truth BoxList
        :param gt_relations: ground truth relationships: np.array (subj_instance_id, obj_instance_id, rel_cate_id)
        :param pred_boxlist: prediction  BoxList
         the rel predictions has already been sorted in descending.
        :param pred_rel_pair_idx: prediction relationship instances pairs index  np.array (n, 2)
        :param pred_rel_scores: prediction relationship predicate scores  np.array  (n, )
        :param eval_rel_pair_prop: prediction relationship instance pair proposals  Top 2048 for for top100 selection
        :return:
        """

        # store the hit index between the ground truth and predictions
        hit_idx = {"rel_hit": [], "pair_det_hit": [], "pair_loc_hit": [], "pred_cls_hit": []}

        if self.eval_rel_pair_prop:
            hit_idx["prop_pair_det_hit"] = []
            hit_idx["prop_pair_loc_hit"] = []

        device = torch.zeros((1, 1)).cpu().device  # cpu_device

        iou_thres = global_container["iou_thres"]

        # transform every array to tensor for adapt the previous code
        # (num_rel, 3) = subj_id, obj_id, rel_labels
        pred_rels = torch.from_numpy(
            np.column_stack((pred_rel_pair_idx, 1 + pred_rel_scores[:, 1:].argmax(1)))
        )
        # (num_rel, )

        instance_hit_iou = boxlist_iou(pred_boxlist, gt_boxlist, to_cuda=False)
        instance_hit_iou = instance_hit_iou.to(device)
        if len(instance_hit_iou) == 0:
            # todo add zero to final results
            pass

        # box pair location hit
        # check the locate results
        inst_loc_hit_idx = instance_hit_iou >= iou_thres
        # (N, 2) array, indicate the which pred box idx matched which gt box idx
        inst_loc_hit_idx = inst_loc_hit_idx.nonzero(as_tuple = False)
        pred_box_loc_hit_idx = inst_loc_hit_idx[:, 0]
        gt_box_loc_hit_idx = inst_loc_hit_idx[:, 1]

        # store the pred box idx hit gt box idx set:
        # the box prediction and gt box are N to M relation,
        # which means one box prediction may hit multiple gt box,
        # so we need to store the each pred box hit gt boxes in set()
        loc_box_matching_results = defaultdict(set)  # key: pred-box index, val: gt-box index
        for each in inst_loc_hit_idx:
            loc_box_matching_results[each[0].item()].add(each[1].item())

        # base on the location results, check the classification results
        gt_det_label_to_cmp = gt_boxlist.get_field("labels")[gt_box_loc_hit_idx]
        pred_det_label_to_cmp = pred_boxlist.get_field("pred_labels")[pred_box_loc_hit_idx]

        # todo working on category clustering later
        if self.instance_class_clustering:
            gt_det_label_to_cmp = copy.deepcopy(gt_det_label_to_cmp)
            pred_det_label_to_cmp = copy.deepcopy(pred_det_label_to_cmp)
            pred_det_label_to_cmp, gt_det_label_to_cmp = trans_cluster_label(
                pred_det_label_to_cmp, gt_det_label_to_cmp, ENTITY_CLUSTER
            )

        pred_det_hit_stat = pred_det_label_to_cmp == gt_det_label_to_cmp

        pred_box_det_hit_idx = pred_box_loc_hit_idx[pred_det_hit_stat]
        gt_box_det_hit_idx = gt_box_loc_hit_idx[pred_det_hit_stat]

        self.result_dict[f"{mode}_instances_det_recall"].append(
            len(torch.unique(gt_box_det_hit_idx)) / (len(gt_boxlist) + 0.000001)
        )
        self.result_dict[f"{mode}_instances_loc_recall"].append(
            len(torch.unique(gt_box_loc_hit_idx)) / (len(gt_boxlist) + 0.000001)
        )
        # store the detection results in matching dict
        det_box_matching_results = defaultdict(set)
        for idx in range(len(pred_box_det_hit_idx)):
            det_box_matching_results[pred_box_det_hit_idx[idx].item()].add(
                gt_box_det_hit_idx[idx].item()
            )

        # after the entities detection recall check, then the entities pairs locating classifications check
        def get_entities_pair_locating_n_cls_hit(to_cmp_pair_mat):
            # according to the detection box hit results,
            # check the location and classification hit of entities pairs
            # instances box location hit res
            rel_loc_pair_mat, rel_loc_init_pred_idx = dump_hit_indx_dict_to_tensor(
                to_cmp_pair_mat, loc_box_matching_results
            )
            # instances box location and category hit
            rel_det_pair_mat, rel_det_init_pred_idx = dump_hit_indx_dict_to_tensor(
                to_cmp_pair_mat, det_box_matching_results
            )
            rel_pair_mat = copy.deepcopy(rel_det_pair_mat)
            rel_init_pred_idx = copy.deepcopy(rel_det_init_pred_idx)

            # use the intersect operate to calculate how the prediction relation pair hit the gt relationship
            # pairs,
            # first is the box pairs location hit and detection hit separately
            rel_loc_hit_idx = (
                intersect_2d_torch_tensor(rel_loc_pair_mat, gt_relations[:, :2])
                .nonzero(as_tuple=False)
                .transpose(1, 0)
            )
            # the index of prediction box hit the ground truth
            pred_rel_loc_hit_idx = rel_loc_init_pred_idx[rel_loc_hit_idx[0]]
            gt_rel_loc_hit_idx = rel_loc_hit_idx[1]  # the prediction hit ground truth index

            rel_det_hit_idx = (
                intersect_2d_torch_tensor(rel_det_pair_mat, gt_relations[:, :2])
                .nonzero(as_tuple=False)
                .transpose(1, 0)
            )
            pred_rel_det_hit_idx = rel_det_init_pred_idx[rel_det_hit_idx[0]]
            gt_rel_det_hit_idx = rel_det_hit_idx[1]

            return (
                rel_loc_pair_mat,
                rel_loc_init_pred_idx,
                rel_pair_mat,
                rel_init_pred_idx,
                pred_rel_loc_hit_idx,
                gt_rel_loc_hit_idx,
                pred_rel_det_hit_idx,
                gt_rel_det_hit_idx,
            )

        # check relation proposal recall
        if self.eval_rel_pair_prop:
            # before relationship rpn
            # prop_rel_pair_mat, prop_rel_init_pred_idx, \
            # prop_rel_loc_hit_idx, prop_rel_loc_hit_gt_idx, \
            # prop_rel_det_hit_idx, prop_rel_det_hit_gt_idx = get_entities_pair_locating_n_cls_hit(rel_pair_prop.pair_mat)
            # rel_proposal_pair_loc_hit_cnt_before_rpn = len(torch.unique(prop_rel_loc_hit_gt_idx))
            # rel_proposal_pair_det_hit_cnt_before_rpn = len(torch.unique(prop_rel_det_hit_gt_idx))

            # after relationship rpn
            (
                prop_rel_loc_pair_mat,
                prop_rel_loc_init_pred_idx,
                prop_rel_pair_mat,
                prop_rel_init_pred_idx,
                prop_rel_loc_hit_idx,
                prop_rel_loc_hit_gt_idx,
                prop_rel_det_hit_idx,
                prop_rel_det_hit_gt_idx,
            ) = get_entities_pair_locating_n_cls_hit(pred_rel_pair_idx)

            rel_proposal_pair_loc_hit_cnt_after_rpn = len(
                torch.unique(prop_rel_loc_hit_gt_idx)
            )
            rel_proposal_pair_det_hit_cnt_after_rpn = len(
                torch.unique(prop_rel_det_hit_gt_idx)
            )

            # self.rel_recall_per_img[topk_idx]['rel_prop_pair_loc_before_relrpn'] \
            #     .append(rel_proposal_pair_loc_hit_cnt_before_rpn / (float(gt_relations.shape[0]) + 0.00001))
            # self, .rel_recall_per_img[topk_idx]['rel_prop_pair_det_before_relrpn'] \
            #     .append(rel_proposal_pair_det_hit_cnt_before_rpn / (float(gt_relations.shape[0]) + 0.00001))
            self.result_dict[f"{mode}_{self.type}_rel_prop_pair_loc_after_relrpn"].append(
                rel_proposal_pair_loc_hit_cnt_after_rpn
                / (float(gt_relations.shape[0]) + 0.00001)
            )
            self.result_dict[f"{mode}_{self.type}_rel_prop_pair_det_after_relrpn"].append(
                rel_proposal_pair_det_hit_cnt_after_rpn
                / (float(gt_relations.shape[0]) + 0.00001)
            )
            self.result_dict[f"{mode}_{self.type}_effective_union_pairs_rate"].append(
                len(prop_rel_loc_hit_idx) / (float(pred_rel_pair_idx.shape[0]) + 0.00001)
            )
            if len(prop_rel_loc_hit_idx) > 0:
                self.result_dict[f"{mode}_{self.type}_effective_union_pairs_range"].append(
                    np.percentile(prop_rel_loc_hit_idx, 95)
                )
            else:
                self.result_dict[f"{mode}_{self.type}_effective_union_pairs_range"].append(
                    self.eval_rel_pair_prop
                )

        # eval the relness and pred clser ranking performance for postive samples

        def eval_roc(scores, matching_results, roc_pred_range=300):
            ref_labels = torch.zeros_like(scores)
            ref_labels[matching_results] = 1

            val, sort_idx = torch.sort(scores, descending=True)
            y = ref_labels[sort_idx[:roc_pred_range]].detach().long().cpu().numpy()
            pred = scores[sort_idx[:roc_pred_range]].detach().cpu().numpy()

            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            roc_res = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}
            return roc_res

        def eval_ap(pred, matched_idx, gt_idx, total_gt_num, pred_range=300):
            # tp + fn

            posb_tp = torch.ones(pred.shape[0], dtype=torch.long) * -1
            posb_tp[matched_idx] = gt_idx
            pred_score, pred_idx = torch.sort(pred, descending=True)

            pred_idx = pred_idx[:pred_range]
            pred_score = pred_score[:pred_range]

            pr_s = []
            recs = []

            for thres in range(1, 10):
                thres *= 0.1
                all_p_idx = pred_score > thres
                all_p_idx = pred_idx[all_p_idx]

                tp_idx = posb_tp >= 0
                mask = torch.zeros(tp_idx.shape[0], dtype=torch.bool)
                mask[all_p_idx] = True
                tp_idx = tp_idx & mask

                tp = len(torch.unique(posb_tp[tp_idx]))

                fp_idx = posb_tp < 0
                mask = torch.zeros(fp_idx.shape[0], dtype=torch.bool)
                mask[all_p_idx] = True
                fp_idx = fp_idx & mask

                fp = len(torch.unique(posb_tp[fp_idx]))

                pr = tp / (tp + fp + 0.0001)
                rec = tp / (total_gt_num + 0.0001)

                pr_s.append(pr)
                recs.append(rec)

            def get_ap(rec, prec):
                """Compute AP given precision and recall."""
                # correct AP calculation
                # first append sentinel values at the end
                mrec = np.concatenate(([0.0], rec, [1.0]))
                mpre = np.concatenate(([0.0], prec, [0.0]))

                # compute the precision envelope
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
                return ap

            return get_ap(np.array(recs), np.array(pr_s))

        # if self.rel_pn_on:
        #     relness_score = pred_boxlist.get_field("relness")
        #     for i in range(relness_score.shape[-1]):

        #         roc_res = eval_roc(
        #             relness_score[:, i], prop_rel_loc_hit_idx, self.vaild_rel_prop_num
        #         )

        #         ap_res = eval_ap(
        #             relness_score[:, i],
        #             prop_rel_loc_hit_idx,
        #             prop_rel_loc_hit_gt_idx,
        #             float(gt_relations.shape[0]),
        #             self.vaild_rel_prop_num,
        #         )

        #         auc = roc_res["auc"]

        #         self.result_dict[
        #             f"{mode}_{self.type}_rel_pn_ap-iter{i}-top{self.vaild_rel_prop_num}"
        #         ].append(ap_res)

        #         if not np.isnan(auc):
        #             self.result_dict[
        #                 f"{mode}_{self.type}_rel_pn_auc-iter{i}-top{self.vaild_rel_prop_num}"
        #             ].append(auc)

        #         roc_res = eval_roc(relness_score[:, i], prop_rel_loc_hit_idx, 100)
        #         ap_res = eval_ap(
        #             relness_score[:, i],
        #             prop_rel_loc_hit_idx,
        #             prop_rel_loc_hit_gt_idx,
        #             float(gt_relations.shape[0]),
        #             100,
        #         )
        #         auc = roc_res["auc"]

        #         self.result_dict[f"{mode}_{self.type}_rel_pn_ap-iter{i}-top100"].append(ap_res)

        #         if not np.isnan(auc):
        #             self.result_dict[f"{mode}_{self.type}_rel_pn_auc-iter{i}-top100"].append(
        #                 auc
        #             )

        # for different top-K relationship filtering, check the recall
        for topk_idx, topk in enumerate((20, 50, 100)):
            selected_rel_pred = pred_rels[:topk]
            # count the detection recall
            # instance_det_hit_num[topk_idx] += len(torch.unique(gt_box_det_hit_idx))
            # instance_det_recall_per_img[topk_idx] \
            #     .append(len(torch.unique(gt_box_det_hit_idx)) / (len(gt_boxes)))

            # after collect the pred box hit result,
            # now need to check the hit of each triplets in gt rel set
            (
                rel_loc_pair_mat,
                rel_loc_init_pred_idx,
                rel_pair_mat,
                rel_init_pred_idx,
                pred_rel_loc_hit_idx,
                gt_rel_loc_hit_idx,
                pred_rel_det_hit_idx,
                gt_rel_det_hit_idx,
            ) = get_entities_pair_locating_n_cls_hit(selected_rel_pred[:, :2])

            if topk == 100:
                pred_rel_scores = pred_boxlist.get_field("pred_rel_scores")
                rel_scores, rel_class = pred_rel_scores[:, 1:].max(dim=1)
                det_score = pred_boxlist.get_field("pred_scores")
                pairs = pred_boxlist.get_field("rel_pair_idxs").long()

                rel_scores_condi_det = (
                    rel_scores * det_score[pairs[:, 0]] * det_score[pairs[:, 1]]
                )
                rel_scores_condi_det = rel_scores_condi_det[:topk]

                # if not torch.isnan(rel_scores_condi_det).any(): # Nan이 뜨는 경우가 존재
                #     roc_res = eval_roc(rel_scores_condi_det, pred_rel_loc_hit_idx, topk)
                #     if not np.isnan(roc_res["auc"]):
                #         self.result_dict[f"{mode}_{self.type}_pred_cls_auc-top{topk}"].append(
                #             roc_res["auc"]
                #         )

            # then we evaluate the full relationship triplets, sub obj detection and predicates
            rel_predicate_label = copy.deepcopy(selected_rel_pred[:, -1][rel_init_pred_idx])
            rel_loc_pair_pred_label = copy.deepcopy(
                selected_rel_pred[:, -1][rel_loc_init_pred_idx]
            )

            def predicates_category_clustering(pred_labels):
                gt_pred_labels = copy.deepcopy(gt_relations[:, -1])
                rel_predicate_label, gt_pred_labels = trans_cluster_label(
                    pred_labels, gt_pred_labels, PREDICATE_CLUSTER
                )
                to_cmp_gt_relationships = copy.deepcopy(gt_relations)
                to_cmp_gt_relationships[:, -1] = gt_pred_labels
                return rel_predicate_label, to_cmp_gt_relationships

            to_cmp_gt_relationships = gt_relations
            if self.predicate_class_clustering:
                (
                    rel_loc_pair_pred_label,
                    to_cmp_gt_relationships,
                ) = predicates_category_clustering(rel_loc_pair_pred_label)
                rel_predicate_label, to_cmp_gt_relationships = predicates_category_clustering(
                    rel_predicate_label
                )

            rel_predicate_label.unsqueeze_(1)

            # eval relationship detection (entities + predicates)
            rel_pair_mat = torch.cat((rel_pair_mat, rel_predicate_label), dim=1)
            rel_hit_idx = (
                intersect_2d_torch_tensor(rel_pair_mat, to_cmp_gt_relationships)
                .nonzero(as_tuple=False)
                .transpose(1, 0)
            )
            pred_rel_hit_idx = rel_init_pred_idx[rel_hit_idx[0]]
            gt_rel_hit_idx = rel_hit_idx[1]

            # eval relationship predicate classification (entities pair loc + predicates)

            rel_loc_pair_pred_label.unsqueeze_(1)
            pred_cls_matrix = torch.cat((rel_loc_pair_mat, rel_loc_pair_pred_label), dim=1)
            pred_cls_hit_idx = (
                intersect_2d_torch_tensor(pred_cls_matrix, to_cmp_gt_relationships)
                .nonzero(as_tuple=False)
                .transpose(1, 0)
            )
            pred_predicate_cls_hit_idx = rel_loc_init_pred_idx[pred_cls_hit_idx[0]]
            gt_pred_cls_hit_idx = pred_cls_hit_idx[1]

            # statistic the prediction results
            # per-class recall
            def stat_per_class_recall_hit(self, hit_type, gt_hit_idx):
                gt_rel_labels = gt_relations[:, -1]
                hit_rel_class_id = gt_rel_labels[gt_hit_idx]
                per_cls_rel_hit = torch.zeros(
                    (cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES, 2), dtype=torch.int64
                )
                # first one is pred hit num, second is gt num
                per_cls_rel_hit[hit_rel_class_id, 0] += 1
                per_cls_rel_hit[gt_rel_labels, 1] += 1
                self.relation_per_cls_hit_recall[hit_type][topk_idx] += per_cls_rel_hit
                self.per_img_rel_cls_recall[topk_idx][hit_type].append(per_cls_rel_hit)

            stat_per_class_recall_hit(self, "rel_hit", gt_rel_hit_idx)
            stat_per_class_recall_hit(self, "pair_loc", gt_rel_loc_hit_idx)
            stat_per_class_recall_hit(self, "pair_det", gt_rel_det_hit_idx)
            stat_per_class_recall_hit(self, "pred_cls", gt_pred_cls_hit_idx)

            # pre image relationship pairs hit counting
            rel_hit_cnt = len(torch.unique(gt_rel_hit_idx))
            pair_det_hit_cnt = len(torch.unique(gt_rel_det_hit_idx))
            pred_cls_hit_cnt = len(torch.unique(gt_pred_cls_hit_idx))
            pair_loc_hit_cnt = len(torch.unique(gt_rel_loc_hit_idx))

            self.result_dict[f"{mode}_{self.type}_pair_loc"][topk].append(
                pair_loc_hit_cnt / (float(gt_relations.shape[0]) + 0.00001)
            )
            self.result_dict[f"{mode}_{self.type}_pair_det"][topk].append(
                pair_det_hit_cnt / (float(gt_relations.shape[0]) + 0.00001)
            )
            self.result_dict[f"{mode}_{self.type}_rel_hit"][topk].append(
                rel_hit_cnt / (float(gt_relations.shape[0]) + 0.00001)
            )
            self.result_dict[f"{mode}_{self.type}_pred_cls"][topk].append(
                pred_cls_hit_cnt / (float(gt_relations.shape[0]) + 0.00001)
            )
