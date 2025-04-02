#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# 定义要遍历的 num_steps 和 inject 值
num_steps_array=(30)
inject_array=(5)
guidance_array=(3)

# 遍历 num_steps 和 inject
for num_steps in "${num_steps_array[@]}"; do
    for inject in "${inject_array[@]}"; do
        for guidance in "${guidance_array[@]}"; do
            echo "Running with num_steps=$num_steps and inject=$inject"
            
            CUDA_VISIBLE_DEVICES=0 python edit.py \
                --source_prompt "a painting of a dog in the forest" \
                --target_prompt "a painting of the forest" \
                --guidance "$guidance" \
                --source_img_dir "/mnt/nas_ssd_cache/434_datasets_ssd/PIE-Bench_v1/annotation_images/3_delete_object_80/1_artificial/4_outdoor/314000000007.jpg"\
                --num_steps "$num_steps" \
                --offload \
                --inject "$inject" \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'rf_zhuzh' \
                --editing_strategy 'replace_ci_ic_cc' \
                --output_prefix "multi_words_edit" \
                --output_dir 'examples/edit-result/try/'
        done
    done
done