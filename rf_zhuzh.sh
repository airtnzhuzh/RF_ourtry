#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# 定义要遍历的 num_steps 和 inject 值
num_steps_array=(20)
inject_array=(2)
guidance_array=(2)

# 遍历 num_steps 和 inject
for num_steps in "${num_steps_array[@]}"; do
    for inject in "${inject_array[@]}"; do
        for guidance in "${guidance_array[@]}"; do
            echo "Running with num_steps=$num_steps and inject=$inject"
            
            CUDA_VISIBLE_DEVICES=0 python edit.py \
                --source_prompt "A young boy is playing with a red toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --target_prompt "A young girl is playing with a blue toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --guidance "$guidance" \
                --source_img_dir 'examples/source/boy.jpg' \
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