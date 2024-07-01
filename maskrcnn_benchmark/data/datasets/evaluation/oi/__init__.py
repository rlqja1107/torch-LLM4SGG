import copy
import os

import numpy as np

from .oi_evaluation import eval_rel_results, eval_entites_detection, eval_classic_recall
#from ..vg.vg_eval import save_output


def oi_evaluation(
        cfg,
        dataset,
        predictions,
        output_folder,
        logger,
        iou_types,
        **_
):
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    result_str = '\n' + '=' * 100 + '\n'

    result_dict_list_to_log = []

    predicate_cls_list = dataset.ind_to_predicates

    groundtruths = []
    # resize predition to same scale with the images
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        # recover original size which is before transform
        predictions[image_id] = prediction.resize((image_width, image_height))
        gt = dataset.get_groundtruth(image_id, evaluation=True)
        groundtruths.append(gt)

    #if cfg.TEST.SAVE_RESULT:
    #    save_output(output_folder, groundtruths, predictions, dataset)

    # eval detection by coco style eval
    if "bbox" in iou_types:
        result_str_tmp = ''
        (mAp,
         result_dict_list_to_log,
         result_str_tmp) = eval_entites_detection(mode, groundtruths, dataset, predictions,
                                              result_dict_list_to_log, result_str_tmp, logger)
        result_str += result_str_tmp
        logger.info(result_str_tmp)

        if not cfg.MODEL.RELATION_ON:
            return mAp, result_dict_list_to_log

    result_str_tmp = ''
    result_str_tmp, \
    result_dict_list_to_log = eval_classic_recall(mode, groundtruths, predictions, predicate_cls_list,
                                                  logger, result_str_tmp, result_dict_list_to_log)
    result_str += result_str_tmp
    logger.info(result_str_tmp)


    # transform the initial prediction into oi predition format
    packed_results = adapt_results(groundtruths, predictions)

    result_str_tmp = ''
    result_str_tmp, \
    result_dict = eval_rel_results(
        packed_results, predicate_cls_list, result_str_tmp, logger,
    )
    result_dict_list_to_log.append(result_dict)

    result_str += result_str_tmp
    logger.info(result_str_tmp)

    logger.info(result_str)
    
    if output_folder:
        with open(os.path.join(output_folder, "evaluation_res.txt"), 'w') as f:
            f.write(result_str)


    return float(result_dict['w_final_score']), result_dict_list_to_log


def adapt_results(
        groudtruths, predictions,
):
    packed_results = []
    for gt, pred in zip(groudtruths, predictions):
        gt = copy.deepcopy(gt)
        pred = copy.deepcopy(pred)

        pred_boxlist = pred.convert('xyxy').to("cpu")
        pred_ent_scores = pred_boxlist.get_field('pred_scores').detach().cpu()
        pred_ent_labels = pred_boxlist.get_field('pred_labels').long().detach().cpu()
        pred_ent_labels = pred_ent_labels - 1  # remove the background class

        pred_rel_pairs = pred_boxlist.get_field('rel_pair_idxs').long().detach().cpu()  # N * R * 2
        pred_rel_scores = pred_boxlist.get_field('pred_rel_scores').detach().cpu()  # N * C

        sbj_boxes = pred_boxlist.bbox[pred_rel_pairs[:, 0], :].numpy()
        sbj_labels = pred_ent_labels[pred_rel_pairs[:, 0]].numpy()
        sbj_scores = pred_ent_scores[pred_rel_pairs[:, 0]].numpy()

        obj_boxes = pred_boxlist.bbox[pred_rel_pairs[:, 1], :].numpy()
        obj_labels = pred_ent_labels[pred_rel_pairs[:, 1]].numpy()
        obj_scores = pred_ent_scores[pred_rel_pairs[:, 1]].numpy()

        prd_scores = pred_rel_scores

        gt_boxlist = gt.convert('xyxy').to("cpu")
        gt_ent_labels = gt_boxlist.get_field('labels')
        gt_ent_labels = gt_ent_labels - 1

        gt_rel_tuple = gt_boxlist.get_field('relation_tuple').long().detach().cpu()
        sbj_gt_boxes = gt_boxlist.bbox[gt_rel_tuple[:, 0], :].detach().cpu().numpy()
        obj_gt_boxes = gt_boxlist.bbox[gt_rel_tuple[:, 1], :].detach().cpu().numpy()
        sbj_gt_classes = gt_ent_labels[gt_rel_tuple[:, 0]].long().detach().cpu().numpy()
        obj_gt_classes = gt_ent_labels[gt_rel_tuple[:, 1]].long().detach().cpu().numpy()
        prd_gt_classes = gt_rel_tuple[:, -1].long().detach().cpu().numpy()
        prd_gt_classes = prd_gt_classes - 1

        return_dict = dict(sbj_boxes=sbj_boxes,
                           sbj_labels=sbj_labels.astype(np.int32, copy=False),
                           sbj_scores=sbj_scores,
                           obj_boxes=obj_boxes,
                           obj_labels=obj_labels.astype(np.int32, copy=False),
                           obj_scores=obj_scores,
                           prd_scores=prd_scores,
                           # prd_scores_bias=prd_scores,
                           # prd_scores_spt=prd_scores,
                           # prd_ttl_scores=prd_scores,
                           gt_sbj_boxes=sbj_gt_boxes,
                           gt_obj_boxes=obj_gt_boxes,
                           gt_sbj_labels=sbj_gt_classes.astype(np.int32, copy=False),
                           gt_obj_labels=obj_gt_classes.astype(np.int32, copy=False),
                           gt_prd_labels=prd_gt_classes.astype(np.int32, copy=False))

        packed_results.append(return_dict)

    return packed_results
