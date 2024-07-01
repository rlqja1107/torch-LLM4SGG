export CUDA_VISIBLE_DEVICES="1"
export NUM_GPUS=1
export output_dir="output/COCO/LLM4SGG_gqa"



python tools/train_net.py \
    --task_config configs/vg150/finetune_gqa.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False DATASETS.TRAIN "('cococaption_scene_graph4GQA',)" OUTPUT_DIR ${output_dir} MODEL.RWT False SOLVER.IMS_PER_BATCH 1 SOLVER.CHECKPOINT_PERIOD 5000 SOLVER.VAL_MIN_ITERATION 100000