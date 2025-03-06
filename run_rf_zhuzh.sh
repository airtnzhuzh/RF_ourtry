# Better Instruction Following


# CUDA_VISIBLE_DEVICES=0 /home/zhuzh/.conda/envs/Fireflow/bin/python edit.py  --source_prompt "A male in a gray jacket with medium-length black hair and a large nose and deep eyebrows on a campus with clear white and yellow tiles, houses and trees in the background" \
#                 --target_prompt "A female in a gray jacket with medium-length black hair and a large nose and deep eyebrows on a campus with clear white and yellow tiles, houses and trees in the background" \
#                 --guidance 2 \
#                 --source_img_dir 'examples/source/zyf.png' \
#                 --num_steps 8  \
#                 --inject 1 \
#                 --name 'flux-dev'  \
#                 --offload \
#                 --start_layer_index 0 \
#                 --end_layer_index 37 \
#                 --reuse_v 0 \
                
#                 --editing_strategy 'add_q' \
#                 --sampling_strategy 'fireflow' \
#                 --output_prefix 'rf_zhuzh_replace_ci_ic' \
#                 --output_dir 'examples/edit-result/art/' 

CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "a vivid depiction of Poseidon, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --target_prompt "a vivid depiction of Batman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --guidance 2 \
                --source_img_dir 'examples/source/art.jpg' \
                --num_steps 8  \
                --inject 20 \
                --offload \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --name 'flux-dev'  \
                --reuse_v 0 \
                --editing_strategy 'add_ci_ic' \
                --sampling_strategy 'rf_zhuzh' \
                --output_prefix 'rf_zhuzh_replace_ci_ic' \
                --output_dir 'examples/edit-result/art/' 