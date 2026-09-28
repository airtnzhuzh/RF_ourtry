# RF_ourtry

Research experiments by **Zhihan Zhu** on rectified-flow inversion and text-guided image editing with FLUX. The code explores numerical sampling, attention replacement, and prompt-token alignment for editing real images.

![Image editing examples](https://github.com/user-attachments/assets/bc9caa35-90fb-47cf-86a7-f66319f5e9cc)

## Main components

- `edit.py`: invert a source image and generate an edit using a target prompt.
- `flux/sampling.py`: inversion and denoising routines, including fourth-order Runge–Kutta sampling.
- `flux/math.py` and `flux/modules/`: token alignment and attention operations.
- `reweight.py`: attention reweighting experiments.
- `calculate_scores.py`: reconstruction evaluation with PSNR, SSIM, and LPIPS.
- `rf_zhuzh.sh` and `re_sh/`: example experiment configurations.

## Setup and usage

Use Python 3.10+ and a CUDA-capable GPU. Install the dependencies:

```bash
pip install -r requirements.txt
```

Before running, update the FLUX checkpoint, autoencoder, T5, and CLIP paths in `flux/util.py` to match your local model files. Model weights and evaluation datasets are not included.

Example editing command:

```bash
python edit.py \
  --source_img_dir examples/source/horse.jpg \
  --source_prompt "a horse" \
  --target_prompt "a zebra" \
  --num_steps 30 --guidance 3 \
  --inject 5 --inject_blocks single \
  --start_layer_index 0 --end_layer_index 37 \
  --editing_strategy replace_ci_ic_ii \
  --offload --output_dir output
```

This is an experimental research repository. Shell and evaluation scripts contain local paths that need adjustment. The legacy `gradio_demo.py` also needs updating to match the current sampling API.

## Acknowledgments

Built on FLUX and adapted from FireFlow image-editing code.
