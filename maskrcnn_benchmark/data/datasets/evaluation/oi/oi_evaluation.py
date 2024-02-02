"""
Written by Ji Zhang, 2019
Some functions are adapted from Rowan Zellers
Original source:
https://github.com/rowanz/neural-motifs/blob/master/lib/evaluation/sg_eval.py
"""
from functools import reduce

import ipdb
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps
from .ap_eval_rel import ap_eval, prepare_mAP_dets
from ..coco.coco_eval import COCOResults
from ..vg150.sgg_eval import SGNoGraphConstraintRecall, SGRecall, SGMeanRecall, SGStagewiseRecall
from ..vg150.vg_eval import evaluate_relation_of_one_image

np.set_printoptions(precision=3)


def eval_entites_detection(mode, groundtruths, dataset, predictions, result_dict_to_log, result_str, logger):
    # create a Coco-like object that we can use to evaluate detection!
    anns = []
    for image_id, gt in enumerate(groundtruths):
        labels = gt.get_field('labels').tolist()  # integer
        boxes = gt.bbox.tolist()  # xyxy
        for cls, box in zip(labels, boxes):
            anns.append({
                'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],  # xywh
                'category_id': cls,
                'id': len(anns),
                'image_id': image_id,
                'iscrowd': 0,
            })
    fauxcoco = COCO()
    fauxcoco.dataset = {
        'info': {'description': 'use coco script for vg detection evaluation'},
        'images': [{'id': i} for i in range(len(groundtruths))],
        'categories': [
            {'supercategory': 'person', 'id': i, 'name': name}
            for i, name in enumerate(dataset.ind_to_classes) if name != '__background__'
        ],
        'annotations': anns,
    }
    fauxcoco.createIndex()

    # format predictions to coco-like
    cocolike_predictions = []
    for image_id, prediction in enumerate(predictions):
        box = prediction.convert('xywh').bbox.detach().cpu().numpy()  # xywh
        score = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#objs,)
        label = prediction.get_field('pred_labels').detach().cpu().numpy()  # (#objs,)
        # for predcls, we set label and score to groundtruth
        if mode == 'predcls':
            label = prediction.get_field('labels').detach().cpu().numpy()
            score = np.ones(label.shape[0])
            assert len(label) == len(box)
        image_id = np.asarray([image_id] * len(box))
        cocolike_predictions.append(
            np.column_stack((image_id, box, score, label))
        )
        # logger.info(cocolike_predictions)
    cocolike_predictions = np.concatenate(cocolike_predictions, 0)

    # logger.info("Evaluating bbox proposals")
    # areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
    # res = COCOResults("box_proposal")
    # for limit in [100, 1000]:
    #     for area, suffix in areas.items():
    #         stats = evaluate_box_proposals(
    #             predictions, dataset, dataset.ind_to_classes, dataset.img_info, area=area, limit=limit
    #         )
    #         key = "AR{}@{:d}".format(suffix, limit)
    #         res.results["box_proposal"][key] = stats["ar"].item()
    # logger.info(res)

    # if output_folder:
    #     torch.save(res, os.path.join(output_folder, "box_proposals.pth"))

    # evaluate via coco API
    res = fauxcoco.loadRes(cocolike_predictions)
    coco_eval = COCOeval(fauxcoco, res, 'bbox')
    coco_eval.params.imgIds = list(range(len(groundtruths)))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_res = COCOResults('bbox')
    coco_res.update(coco_eval)
    mAp = coco_eval.stats[1]

    def get_coco_eval(coco_eval, iouThr, eval_type, maxDets=-1, areaRng="all"):
        p = coco_eval.params

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        if maxDets == -1:
            max_range_i = np.argmax(p.maxDets)
            mind = [max_range_i, ]
        else:
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if eval_type == 'precision':
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        elif eval_type == 'recall':
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        else:
            raise ValueError("Invalid eval metrics")
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return p.maxDets[mind[-1]], mean_s

    coco_res_to_save = {}
    for key, value in coco_res.results.items():
        for evl_name, eval_val in value.items():
            coco_res_to_save[f"{key}/{evl_name}"] = eval_val
    result_dict_to_log.append(coco_res_to_save)

    result_str += 'Detection evaluation mAp=%.4f\n' % mAp
    result_str += "recall@%d IOU:0.5 %.4f\n" % get_coco_eval(coco_eval, 0.5, 'recall')
    result_str += '=' * 100 + '\n'
    avg_metrics = mAp
    logger.info(result_str)
    result_str = '\n'
    logger.info("box evaluation done!")

    return avg_metrics, result_dict_to_log, result_str


def eval_rel_results(all_results, predicate_cls_list, result_str, logger):

    logger.info('openimage evaluation: \n')


    topk = 100

    # if cfg.TEST.DATASETS[0].find('vg') >= 0:
    #     eval_per_img = True
    #     # eval_per_img = False
    #     prd_k = 1
    # else:
    #     eval_per_img = False
    #     prd_k = 2
    #
    # if cfg.TEST.DATASETS[0].find('oi') >= 0:
    #     eval_ap = True
    # else:
    #     eval_ap = False

    # here we only takes the evaluation option of openimages
    prd_k = 2

    recalls_per_img = {1: [], 5: [], 10: [], 20: [], 50: [], 100: []}
    recalls = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    all_gt_cnt = 0

    topk_dets = []
    for im_i, res in enumerate(tqdm(all_results)):

        # in oi_all_rel some images have no dets
        if res['prd_scores'] is None:
            det_boxes_s_top = np.zeros((0, 4), dtype=np.float32)
            det_boxes_o_top = np.zeros((0, 4), dtype=np.float32)
            det_labels_s_top = np.zeros(0, dtype=np.int32)
            det_labels_p_top = np.zeros(0, dtype=np.int32)
            det_labels_o_top = np.zeros(0, dtype=np.int32)
            det_scores_top = np.zeros(0, dtype=np.float32)

            det_scores_top_vis = np.zeros(0, dtype=np.float32)
            if 'prd_scores_bias' in res:
                det_scores_top_bias = np.zeros(0, dtype=np.float32)
            if 'prd_scores_spt' in res:
                det_scores_top_spt = np.zeros(0, dtype=np.float32)
        else:
            det_boxes_sbj = res['sbj_boxes']  # (#num_rel, 4)
            det_boxes_obj = res['obj_boxes']  # (#num_rel, 4)
            det_labels_sbj = res['sbj_labels']  # (#num_rel,)
            det_labels_obj = res['obj_labels']  # (#num_rel,)
            det_scores_sbj = res['sbj_scores']  # (#num_rel,)
            det_scores_obj = res['obj_scores']  # (#num_rel,)
            if 'prd_scores_ttl' in res:
                det_scores_prd = res['prd_scores_ttl'][:, 1:]
            else:
                det_scores_prd = res['prd_scores'][:, 1:] # N x C (the prediction score of each categories)


            det_labels_prd = np.argsort(-det_scores_prd, axis=1) # N x C (the prediction labels sort by prediction score)
            det_scores_prd = -np.sort(-det_scores_prd, axis=1) # N x C (the prediction scores sort by prediction score)

            # filtering the results by the productiong of prediction score of subject object and predicates
            det_scores_so = det_scores_sbj * det_scores_obj  # N
            det_scores_spo = det_scores_so[:, None] * det_scores_prd[:, :prd_k] # N x prd_K
            # (take top k predicates prediction of each pairs as final prediction, approximation of non-graph constrain setting)

            det_scores_inds = argsort_desc(det_scores_spo)[:topk] # topk x 2
            # selected the topk score prediction from the N x prd_k predictions
            # first dim: which pair prediction. second dim: which cate prediction from this pair

            # take out the correspond tops relationship predation scores and pair boxes and their labels.
            det_scores_top = det_scores_spo[det_scores_inds[:, 0], det_scores_inds[:, 1]]
            det_boxes_so_top = np.hstack(
                (det_boxes_sbj[det_scores_inds[:, 0]], det_boxes_obj[det_scores_inds[:, 0]]))
            det_labels_p_top = det_labels_prd[det_scores_inds[:, 0], det_scores_inds[:, 1]]
            det_labels_spo_top = np.vstack(
                (det_labels_sbj[det_scores_inds[:, 0]], det_labels_p_top,
                 det_labels_obj[det_scores_inds[:, 0]])).transpose()

            # filter the very low prediction scores relationship prediction
            cand_inds = np.where(det_scores_top > 0.00001)[0]
            det_boxes_so_top = det_boxes_so_top[cand_inds]
            det_labels_spo_top = det_labels_spo_top[cand_inds]
            det_scores_top = det_scores_top[cand_inds]

            # auxlary prediction score, but not use
            det_scores_vis = res['prd_scores'][:, 1:]
            for i in range(det_labels_prd.shape[0]):
                det_scores_vis[i] = det_scores_vis[i][det_labels_prd[i]]
            det_scores_vis = det_scores_vis[:, :prd_k]
            det_scores_top_vis = det_scores_vis[det_scores_inds[:, 0], det_scores_inds[:, 1]]
            det_scores_top_vis = det_scores_top_vis[cand_inds]
            if 'prd_scores_bias' in res:
                det_scores_bias = res['prd_scores_bias'][:, 1:]
                for i in range(det_labels_prd.shape[0]):
                    det_scores_bias[i] = det_scores_bias[i][det_labels_prd[i]]
                det_scores_bias = det_scores_bias[:, :prd_k]
                det_scores_top_bias = det_scores_bias[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                det_scores_top_bias = det_scores_top_bias[cand_inds]
            if 'prd_scores_spt' in res:
                det_scores_spt = res['prd_scores_spt'][:, 1:]
                for i in range(det_labels_prd.shape[0]):
                    det_scores_spt[i] = det_scores_spt[i][det_labels_prd[i]]
                det_scores_spt = det_scores_spt[:, :prd_k]
                det_scores_top_spt = det_scores_spt[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                det_scores_top_spt = det_scores_top_spt[cand_inds]

            det_boxes_s_top = det_boxes_so_top[:, :4]
            det_boxes_o_top = det_boxes_so_top[:, 4:]
            det_labels_s_top = det_labels_spo_top[:, 0]
            det_labels_p_top = det_labels_spo_top[:, 1]
            det_labels_o_top = det_labels_spo_top[:, 2]

        topk_dets.append(dict(image=im_i,
                              det_boxes_s_top=det_boxes_s_top,
                              det_boxes_o_top=det_boxes_o_top,
                              det_labels_s_top=det_labels_s_top,
                              det_labels_p_top=det_labels_p_top,
                              det_labels_o_top=det_labels_o_top,
                              det_scores_top=det_scores_top, )
                         )

        topk_dets[-1]['det_scores_top_vis'] = det_scores_top_vis
        if 'prd_scores_bias' in res:
            topk_dets[-1]['det_scores_top_bias'] = det_scores_top_bias
        if 'prd_scores_spt' in res:
            topk_dets[-1]['det_scores_top_spt'] = det_scores_top_spt

        gt_boxes_sbj = res['gt_sbj_boxes']  # (#num_gt, 4)
        gt_boxes_obj = res['gt_obj_boxes']  # (#num_gt, 4)
        gt_labels_sbj = res['gt_sbj_labels']  # (#num_gt,)
        gt_labels_obj = res['gt_obj_labels']  # (#num_gt,)
        gt_labels_prd = res['gt_prd_labels']  # (#num_gt,)
        gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
        gt_labels_spo = np.vstack((gt_labels_sbj, gt_labels_prd, gt_labels_obj)).transpose()
        # Compute recall. It's most efficient to match once and then do recall after
        # det_boxes_so_top is (#num_rel, 8)
        # det_labels_spo_top is (#num_rel, 3)
        pred_to_gt = _compute_pred_matches(
            gt_labels_spo, det_labels_spo_top,
            gt_boxes_so, det_boxes_so_top)

        # perimage recall
        for k in recalls_per_img:
            if len(pred_to_gt):
                match = reduce(np.union1d, pred_to_gt[:k])
            else:
                match = []
            rec_i = float(len(match)) / float(gt_labels_spo.shape[0] + 1e-12)  # in case there is no gt
            recalls_per_img[k].append(rec_i)

        # all dataset recall
        all_gt_cnt += gt_labels_spo.shape[0]
        for k in recalls:
            if len(pred_to_gt):
                match = reduce(np.union1d, pred_to_gt[:k])
            else:
                match = []
            recalls[k] += len(match)

        topk_dets[-1].update(dict(gt_boxes_sbj=gt_boxes_sbj,
                                  gt_boxes_obj=gt_boxes_obj,
                                  gt_labels_sbj=gt_labels_sbj,
                                  gt_labels_obj=gt_labels_obj,
                                  gt_labels_prd=gt_labels_prd))



    for k in recalls_per_img.keys():
        recalls_per_img[k] = np.mean(recalls_per_img[k])

    for k in recalls:
        recalls[k] = float(recalls[k]) / (float(all_gt_cnt) + 1e-12)

    rel_prd_cats = predicate_cls_list[1:]  # remove the background categoires

    # prepare dets for each class
    logger.info('Preparing dets for mAP...')
    cls_image_ids, cls_dets, cls_gts, npos = prepare_mAP_dets(topk_dets, len(rel_prd_cats))
    all_npos = sum(npos)

    rel_mAP = 0.
    w_rel_mAP = 0.
    ap_str = ''
    per_class_res = ''
    for c in range(len(rel_prd_cats)):
        rec, prec, ap = ap_eval(cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], True)
        weighted_ap = ap * float(npos[c]) / float(all_npos)
        w_rel_mAP += weighted_ap
        rel_mAP += ap
        ap_str += '{:.2f}, '.format(100 * ap)
        per_class_res += '{}: {:.3f} / {:.3f} ({:.6f}), '.format(
            rel_prd_cats[c], 100 * ap, 100 * weighted_ap, float(npos[c]) / float(all_npos))

    rel_mAP /= len(rel_prd_cats)
    result_str += '\nrel mAP: {:.2f}, weighted rel mAP: {:.2f}\n'.format(100 * rel_mAP, 100 * w_rel_mAP)
    result_str += 'rel AP perclass: AP/ weighted-AP (recall)\n'
    result_str += per_class_res + "\n\n"
    phr_mAP = 0.
    w_phr_mAP = 0.
    ap_str = ''

    per_class_res = ''
    for c in range(len(rel_prd_cats)):
        rec, prec, ap = ap_eval(cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], False)
        weighted_ap = ap * float(npos[c]) / float(all_npos)
        w_phr_mAP += weighted_ap
        phr_mAP += ap
        ap_str += '{:.2f}, '.format(100 * ap)
        per_class_res += '{}: {:.3f} / {:.3f} ({:.6f}), '.format(
            rel_prd_cats[c], 100 * ap, 100 * weighted_ap, float(npos[c]) / float(all_npos))

    phr_mAP /= len(rel_prd_cats)
    result_str += '\nphr mAP: {:.2f}, weighted phr mAP: {:.2f}\n'.format(100 * phr_mAP, 100 * w_phr_mAP)
    result_str += 'phr AP perclass: AP/ weighted-AP (recall)\n'
    result_str += per_class_res + "\n\n"

    # total: 0.4 x rel_mAP + 0.2 x R@50 + 0.4 x phr_mAP
    final_score = 0.4 * rel_mAP + 0.2 * recalls[50] + 0.4 * phr_mAP

    # total: 0.4 x w_rel_mAP + 0.2 x R@50 + 0.4 x w_phr_mAP
    w_final_score = 0.4 * w_rel_mAP + 0.2 * recalls[50] + 0.4 * w_phr_mAP
    result_str += "recall@50: {:.2f}, recall@100: {:.2f}\n".format(100 * recalls[50], 100 * recalls[100])
    result_str += "recall@50: {:.2f}, recall@100: {:.2f} (per images)\n\n".format(100 * recalls_per_img[50],
                                                                                  100 * recalls_per_img[100])

    result_str += "weighted_res: 0.4 * w_rel_mAP + 0.2 * recall@50 + 0.4 * w_phr_mAP \n"
    result_str += 'final_score:{:.2f}  weighted final_score: {:.2f}\n'.format(final_score * 100, w_final_score*100)

    res_dict = dict(
        mAP_rel=rel_mAP,
        wmAP_rel=w_rel_mAP,
        mAP_phr=phr_mAP,
        wmAP_phr=w_phr_mAP,
        R50=recalls[50],
        final_score=final_score,
        w_final_score=w_final_score,
    )

    result_str += "=" * 80
    result_str += "\n\n"

    logger.info('Done.')


    # logger.info(result_str)

    return result_str, res_dict


def eval_classic_recall(mode, groundtruths, predictions, predicate_cls_list,
                        logger, result_str, result_dict_list_to_log):
    evaluator = {}
    rel_eval_result_dict = {}
    eval_recall = SGRecall(rel_eval_result_dict)
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall

    eval_nog_recall = SGNoGraphConstraintRecall(rel_eval_result_dict)
    eval_nog_recall.register_container(mode)
    evaluator['eval_nog_recall'] = eval_nog_recall

    # used for meanRecall@K
    eval_mean_recall = SGMeanRecall(rel_eval_result_dict, len(predicate_cls_list), predicate_cls_list,
                                    print_detail=True)
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall

    eval_stagewise_recall = SGStagewiseRecall(rel_eval_result_dict)
    eval_stagewise_recall.register_container(mode)
    evaluator['eval_stagewise_recall'] = eval_stagewise_recall

    # prepare all inputs
    global_container = {}
    global_container['result_dict'] = rel_eval_result_dict
    global_container['mode'] = mode
    global_container['num_rel_category'] = len(predicate_cls_list)
    global_container['iou_thres'] = cfg.TEST.RELATION.IOU_THRESHOLD
    global_container['attribute_on'] = False

    logger.info("evaluating relationship predictions..")
    for groundtruth, prediction in tqdm(zip(groundtruths, predictions), total=len(predictions)):
        evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)

    # calculate mean recall
    eval_mean_recall.calculate_mean_recall(mode)

    # print result
    result_str += "classic recall evaluations:\n"
    result_str += eval_recall.generate_print_string(mode)
    result_str += eval_nog_recall.generate_print_string(mode)
    result_str += eval_mean_recall.generate_print_string(mode)
    result_str += eval_stagewise_recall.generate_print_string(mode)

    def generate_eval_res_dict(evaluator, mode):
        res_dict = {}
        for k, v in evaluator.result_dict[f'{mode}_{evaluator.type}'].items():
            res_dict[f'{mode}_{evaluator.type}/top{k}'] = np.mean(v)
        return res_dict

    result_dict_list_to_log.extend([generate_eval_res_dict(eval_recall, mode),
                                    generate_eval_res_dict(eval_nog_recall, mode),
                                    generate_eval_res_dict(eval_mean_recall, mode), ])
    result_str += "\n" + "=" * 80 +"\n"

    return result_str, result_dict_list_to_log


# This function is adapted from Rowan Zellers' code:
# https://github.com/rowanz/neural-motifs/blob/master/lib/evaluation/sg_eval.py
# Modified for this project to work with PyTorch v0.4
def _compute_pred_matches(gt_triplets, pred_triplets,
                          gt_boxes, pred_boxes, iou_thresh=0.5, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: Do y
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            gt_box_union = gt_box_union.astype(dtype=np.float32, copy=False)
            box_union = box_union.astype(dtype=np.float32, copy=False)
            inds = bbox_overlaps(gt_box_union[None],
                                 box_union=box_union)[0] >= iou_thresh

        else:
            gt_box = gt_box.astype(dtype=np.float32, copy=False)
            boxes = boxes.astype(dtype=np.float32, copy=False)
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
