export CUDA_VISIBLE_DEVICES="1"
export test_list=("")

export output_dir="output/GQA/"

for t in ${test_list[@]}
do
    python tools/test_net.py --config-file "${output_dir}/config.yml" --weight "${output_dir}/model_${t}.pth"
done
