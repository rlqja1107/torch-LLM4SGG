export CUDA_VISIBLE_DEVICES="0,1" # Choose your GPU number
export NUM_GPUS=2 # Number of GPU
export output_dir="output/CC/LLM4SGG_vg"

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=59996 tools/train_net.py \
    --task_config configs/vg150/finetune_VG.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False DATASETS.TRAIN "('cccaption_scene_graph',)" OUTPUT_DIR ${output_dir} MODEL.RWT False SOLVER.IMS_PER_BATCH 1 SOLVER.CHECKPOINT_PERIOD 5000 SOLVER.VAL_MIN_ITERATION 100000 
