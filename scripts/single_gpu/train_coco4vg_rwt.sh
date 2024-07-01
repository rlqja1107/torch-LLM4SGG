export CUDA_VISIBLE_DEVICES="0"
export NUM_GPUS=1
export output_dir="output/COCO/LLM4SGG_rwt_vg"



python tools/train_net.py \
    --task_config configs/vg150/finetune.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False DATASETS.TRAIN "('cococaption_scene_graph',)" OUTPUT_DIR ${output_dir} MODEL.RWT True SOLVER.IMS_PER_BATCH 1