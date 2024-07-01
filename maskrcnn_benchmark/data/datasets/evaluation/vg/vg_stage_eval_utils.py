import numpy as np
import torch

from maskrcnn_benchmark.utils.miscellaneous import intersect_2d


def boxlist_iou(boxlist1, boxlist2, to_cuda=True):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    if to_cuda:
        if boxlist1.bbox.device.type != 'cuda':
            boxlist1.bbox = boxlist1.bbox.cuda()
        if boxlist2.bbox.device.type != 'cuda':
            boxlist2.bbox = boxlist2.bbox.cuda()

    box1 = boxlist1.bbox
    box2 = boxlist2.bbox

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def intersect_2d_torch_tensor(x1, x2):
    return torch.from_numpy(intersect_2d(x1.numpy(), x2.numpy()))


def dump_hit_indx_dict_to_tensor(pred_pair_mat, gt_box_hit_idx_dict):
    """
    for compare the prediction and gt easily, we need to expand the N to M box match results to
    array.
    here, give relationship prediction pair matrix, expand the gt_box_hit_idx_dit to the array.
    We do the full connection of hit gt box idx of each prediction pairs
    :param pred_pair_mat:
    :param gt_box_hit_idx_dict: the hit gt idx of each prediction box
    :return:
        to_cmp_pair_mat: expanded relationship pair result (N, 2), store the gt box indexs.
            N is large than initial prediction pair matrix
        initial_pred_idx_seg: marking the seg for each pred pairs. If it hit multiple detection gt,
            it could have more than one prediction pairs, we need to mark that they are indicated to
            same initial predations
    """
    to_cmp_pair_mat = []
    initial_pred_idx_seg = []
    # write result into the pair mat
    for pred_idx, pred_pair in enumerate(pred_pair_mat):
        sub_pred_hit_idx_set = gt_box_hit_idx_dict[pred_pair[0].item()]
        obj_pred_hit_idx_set = gt_box_hit_idx_dict[pred_pair[1].item()]
        # expand the prediction index by full combination
        for each_sub_hit_idx in sub_pred_hit_idx_set:
            for each_obj_hit_idx in obj_pred_hit_idx_set:
                to_cmp_pair_mat.append([each_sub_hit_idx, each_obj_hit_idx])
                initial_pred_idx_seg.append(pred_idx)  #
    if len(to_cmp_pair_mat) == 0:
        to_cmp_pair_mat = torch.zeros((0, 2), dtype=torch.int64)
    else:
        to_cmp_pair_mat = torch.from_numpy(np.array(to_cmp_pair_mat, dtype=np.int64))

    initial_pred_idx_seg = torch.from_numpy(np.array(initial_pred_idx_seg, dtype=np.int64))
    return to_cmp_pair_mat, initial_pred_idx_seg


LONGTAIL_CATE_IDS_DICT = {
    'head': [31, 20, 22, 30, 48],
    'body': [29, 50, 1, 21, 8, 43, 40, 49, 41, 23, 7, 6, 19, 33, 16, 38],
    'tail': [11, 14, 46, 37, 13, 24, 4, 47, 5, 10, 9, 34, 3, 25, 17, 35, 42, 27, 12, 28,
             39, 36, 2, 15, 44, 32, 26, 18, 45]
}

LONGTAIL_CATE_IDS_QUERY = {}
for long_name, cate_id in LONGTAIL_CATE_IDS_DICT.items():
    for each_cate_id in cate_id:
        LONGTAIL_CATE_IDS_QUERY[each_cate_id] = long_name

PREDICATE_CLUSTER = [[50, 20, 9], [22, 48, 49], [31], [31, 41, 1], [31, 30]]
ENTITY_CLUSTER = [[91, 149, 53, 78, 20, 79, 90, 56, 68]]


def get_cluster_id(cluster, cate_id):
    for idx, each in enumerate(cluster):
        if cate_id in each:
            return each[0]
    return -1


def transform_cateid_into_cluster_id(cate_list, cluster):
    for idx in range(len(cate_list)):
        cluster_id = get_cluster_id(cluster, cate_list[idx].item())

        if cluster_id != -1:
            cate_list[idx] = cluster_id
    return cate_list


def trans_cluster_label(pred_pred_cate_list, gt_pred_cate_list, cluster):
    """
    transform the categories labels to cluster label for label overlapping avoiding
    :param pred_pair_mat: (subj_id, obj-id, cate-lable)
    :param gt_pair_mat:
    :return:
    """
    cluster_ref_pred_cate = transform_cateid_into_cluster_id(pred_pred_cate_list, cluster)
    cluster_ref_gt_cate = transform_cateid_into_cluster_id(gt_pred_cate_list, cluster)

    return cluster_ref_pred_cate, cluster_ref_gt_cate
