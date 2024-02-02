from .vg_eval import do_vg_evaluation


def vg150_evaluation(
    cfg,
    dataset,
    predictions,
    output_folder,
    iou_types,
    **_
):
    return do_vg_evaluation(
        cfg=cfg,
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        iou_types=iou_types,
    )
