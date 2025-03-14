
# CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
#                 --target_prompt "A young girl is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
#                 --guidance 2 \
#                 --source_img_dir 'examples/source/boy.jpg' \
#                 --num_steps 10 \
#                 --offload \
#                 --inject 1 \
#                 --start_layer_index 0 \
#                 --end_layer_index 37 \
#                 --reuse_v 0 \
#                 --sampling_strategy 'rf_ourtry' \
#                 --editing_strategy 'add_ci_ic_cc' \
#                 --output_prefix 'ourtry_boy' \
#                 --output_dir 'examples/edit-result/boy/' 


#!/bin/bash

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
                --source_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --target_prompt "A young girl is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --guidance "$guidance" \
                --source_img_dir 'examples/source/boy.jpg' \
                --num_steps "$num_steps" \
                --offload \
                --inject "$inject" \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --reuse_v 0 \
                --sampling_strategy 'rf_ourtry' \
                --editing_strategy 'add_ci_ic_cc' \
                --output_prefix "ourtry" \
                --output_dir 'examples/edit-result/try/'
        done
    done
done