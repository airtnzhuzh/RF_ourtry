import json
import os
from PIL import Image
import torch
import argparse
import numpy as np
import lpips
from pycocotools import mask as maskUtils
import os.path as osp
from accelerate.utils import set_seed
from mlcbase import Logger, listdir, load_json, save_json, create
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import random
import edit
def parse_args():
    parser = argparse.ArgumentParser(description="Image Editing with Flux")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save the edited images")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--guidance", type=float, default=1, help="Guidance scale for image generation")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--inject", type=int, default=0, help="Injection step for editing")
    parser.add_argument("--editing_strategy", type=str, default='replace_ci_ic_cc', help="Editing strategy")
    return parser.parse_args()
def clean_prompt(prompt):
    """删除 prompt 中的 [ 和 ]"""
    return prompt.replace("[", "").replace("]", "")

def rle_decode(rle_str, height, width):
    rle_encoding = maskUtils.frPyObjects([rle_str], height, width)
    segmentation_mask = maskUtils.decode(rle_encoding)
    return np.transpose(segmentation_mask, (1, 0, 2))[:, :, 0]
def pil2tensor(image: Image.Image, normalize: bool = False) -> torch.Tensor:
    image = np.array(image).astype(np.float32) / 255.0  # normalize to [0, 1]
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # add batch dimension
    if normalize:
        image = 2.0 * image - 1.0  # normalize to [-1, 1]
    return image
def main(): 
    args = parse_args()


    if not osp.exists(args.save_dir):
        create(args.save_dir, "dir")
        create(osp.join(args.save_dir, "images"), "dir")
    logger = Logger()
    logger.init_logger(osp.join(args.save_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)
    
    args_dict = vars(args)
    args_text = "Arguments:\n"
    for k, v in args_dict.items():
        args_text += f"{k}: {v}\n"
    logger.info(args_text)


   #load mapping
    mapping = load_json(args.data_dir, "mapping_file.json")
    lpips_loss = lpips.LPIPS(net="alex").to(args.device)
    results = {"mean_psnr": None, "mean_ssim": None, "mean_lpips": None, "mean_elapsed": None, "images": []}
    total_psnr = 0.
    total_ssim = 0.
    total_lpips = 0.
    total_elapsed = 0.
    # 遍历所有样本
    for sample_id, ann in mapping.items():
        # 1. 加载原始图片
        image_path = osp.join(args.data_dir+"annotation_images"+ann["image_path"])
        ori_image = Image.open(image_path).convert("RGB")
        
        # 2. 清理 prompt（删除 [ 和 ]）
        source_prompt = clean_prompt(ann["original_prompt"])
        target_prompt = clean_prompt(ann["editing_prompt"])
        mask = rle_decode(ann["mask"], ori_image.size[1], ori_image.size[0])
        mask = np.array(mask, dtype=np.uint8)


        
        # 3. 调用编辑函数
        recon_image, t1, t0 = edit.rf_ourtry(
            init_image=ori_image,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            guidance=args.guidance,
            num_steps=args.num_steps,
            inject=args.inject,
            editing_strategy=args.editing_strategy,
            seed=args.seed
        )

        psnr_score = psnr(np.array(ori_image), np.array(recon_image))
        ssim_score = ssim(np.array(ori_image), np.array(recon_image), win_size=7, channel_axis=2)
        lpips_score = lpips_loss(pil2tensor(ori_image).to(args.device), pil2tensor(recon_image).to(args.device)).item()
        
        # 4. 保存结果图片
        output_path = os.path.join(args.output_dir, f"{sample_id}.jpg")
        recon_image.save(output_path)
        print(f"Saved edited image to {output_path}")
        elapsed = t1 - t0
        logger.success(f"PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}, LPIPS: {lpips_score:.4f}, Elapsed: {elapsed:.4f}")
        results["images"].append({
            "sample_id": ann["sample_id"],
            "editing_type_id": ann["editing_type_id"],
            "image_path": output_path,
            "psnr": psnr_score,
            "ssim": ssim_score,
            "lpips": lpips_score,
            "elapsed": elapsed,
            "width": ori_image.width,
            "height": ori_image.height,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "editing_instruction": ann["editing_instruction"],
        })
        total_psnr += psnr_score
        total_ssim += ssim_score
        total_lpips += lpips_score
        total_elapsed += elapsed
        results["mean_psnr"] = total_psnr / len(results["images"])
        results["mean_ssim"] = total_ssim / len(results["images"])
        results["mean_lpips"] = total_lpips / len(results["images"])
        results["mean_elapsed"] = total_elapsed / len(results["images"])

        save_json(results, osp.join(args.save_dir, "results.json"))
        recon_image.save(osp.join(args.save_dir, "images", ann["image"]))

if __name__ == "__main__":
    main()