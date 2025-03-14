import os
import json
import re
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms

# 初始化配置
source_dir = '/home/zhuzh/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/Kodak'  # 源图像目录
result_dir = '/home/zhuzh/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/src/examples/edit-result/reconstruction/Step30_rf4'          # 结果图像目录
output_json = '/home/zhuzh/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/src/re_sh/inversion_reconstruction_diff_steps_30_rk4'   # 输出文件

# 初始化LPIPS模型
loss_fn = lpips.LPIPS(net='alex').eval()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 图像转换器
transform = transforms.Compose([
    transforms.ToTensor()
])

def load_image_pair(src_path, res_path):
    """加载并验证图像对"""
    try:
        src_img = Image.open(src_path).convert('RGB')
        res_img = Image.open(res_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading images: {e}")

    if src_img.size != res_img.size:
        raise ValueError(f"Image size mismatch: {src_img.size} vs {res_img.size}")
    
    return src_img, res_img

def calculate_metrics(src_img, res_img):
    """计算所有指标"""
    # 转换为numpy数组
    src_np = np.array(src_img)
    res_np = np.array(res_img)
    
    # 计算SSIM，只返回SSIM值
    ssim_value = ssim(src_np, res_np, 
                     data_range=255, 
                     channel_axis=2,
                     win_size=11,
                     gaussian_weights=True,
                     full=True)[0]  # 确保只返回SSIM值
    
    # 计算PSNR
    psnr_value = psnr(src_np, res_np, data_range=255)
    
    # 转换为Tensor并计算LPIPS
    src_tensor = transform(src_img).unsqueeze(0)
    res_tensor = transform(res_img).unsqueeze(0)
    
    if torch.cuda.is_available():
        src_tensor = src_tensor.cuda()
        res_tensor = res_tensor.cuda()
    
    with torch.no_grad():
        lpips_value = loss_fn(src_tensor, res_tensor).item()
    
    return {
        'SSIM': float(ssim_value),  # 现在ssim_value是单个浮点数
        'PSNR': float(psnr_value),
        'LPIPS': float(lpips_value)
    }

# 主处理流程
results = {}
for filename in os.listdir(result_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    # 提取kodim编号
    match = re.search(r'kodim(\d{2})', filename)
    if not match:
        print(f"跳过无法识别的文件: {filename}")
        continue
    
    kodim_num = match.group(1)
    src_path = os.path.join(source_dir, f'kodim{kodim_num}.png')  # 假设源文件是PNG格式
    
    if not os.path.exists(src_path):
        print(f"源文件不存在: {src_path}")
        continue
    
    try:
        src_img, res_img = load_image_pair(src_path, os.path.join(result_dir, filename))
        metrics = calculate_metrics(src_img, res_img)
        results[filename] = metrics
        print(f"处理完成: {filename} - SSIM: {metrics['SSIM']:.4f}")
    except Exception as e:
        print(f"处理 {filename} 时出错: {str(e)}")
        continue

# 计算平均值
avg_metrics = {
    'SSIM': np.mean([v['SSIM'] for v in results.values()]),
    'PSNR': np.mean([v['PSNR'] for v in results.values()]),
    'LPIPS': np.mean([v['LPIPS'] for v in results.values()])
}
results['average'] = avg_metrics

# 保存结果
with open(output_json, 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n处理完成，结果已保存至 {output_json}")
print(f"平均指标:")
print(f"SSIM:  {avg_metrics['SSIM']:.4f}")
print(f"PSNR:  {avg_metrics['PSNR']:.2f} dB")
print(f"LPIPS: {avg_metrics['LPIPS']:.4f}")