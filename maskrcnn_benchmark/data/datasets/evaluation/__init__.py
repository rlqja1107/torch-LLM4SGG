from maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation
from .vg import vg_evaluation
from .vg150 import vg150_evaluation
from .box_aug import im_detect_bbox_aug
from .od_to_grounding import od_to_grounding_evaluation
from .oi import oi_evaluation
from .gqa import gqa_evaluation
from maskrcnn_benchmark.utils.logger import setup_logger
def evaluate(dataset, predictions, output_folder, config=None, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset) or isinstance(dataset, datasets.TSVDataset):
        return coco_evaluation(**args)
    # elif isinstance(dataset, datasets.VGTSVDataset):
    #     return vg_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.CocoDetectionTSV):
        return od_to_grounding_evaluation(**args)
    elif isinstance(dataset, datasets.OIDataset):
        return oi_evaluation(**args)
    elif isinstance(dataset, datasets.VG150Dataset):
        args['cfg'] = config
        return vg150_evaluation(**args)
    elif isinstance(dataset, datasets.LvisDetection):
        pass
    elif isinstance(dataset, datasets.GQADataset):
        args['cfg'] = config
        return gqa_evaluation(**args)

    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))


def evaluate_mdetr(dataset, predictions, output_folder, cfg):
   
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset) or isinstance(dataset, datasets.TSVDataset):
        return coco_evaluation(**args)
    # elif isinstance(dataset, datasets.VGTSVDataset):
    #     return vg_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.CocoDetectionTSV):
        return od_to_grounding_evaluation(**args)
    elif isinstance(dataset, datasets.LvisDetection):
        pass
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
