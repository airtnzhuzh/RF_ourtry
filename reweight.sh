#!/bin/bash

# 定义要遍历的 num_steps 和 inject 值
num_steps_array=(20)
inject_array=(4)
guidance_array=(2)

# 遍历 num_steps 和 inject
for num_steps in "${num_steps_array[@]}"; do
    for inject in "${inject_array[@]}"; do
        for guidance in "${guidance_array[@]}"; do
            echo "Running with num_steps=$num_steps and inject=$inject"
            
            CUDA_VISIBLE_DEVICES=0 python reweight.py \
                --source_prompt "The boulevards are crowded today." \
                --guidance "$guidance" \
                --source_img_dir 'examples/source/crowd.png' \
                --num_steps "$num_steps" \
                --offload \
                --inject "$inject" \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'rf_zhuzh' \
                --reweight_word "crowded"\
                --output_prefix "ourtry" \
                --output_dir 'examples/edit-result/try/' \
                --reweight_times 0.6
        done
    done
done