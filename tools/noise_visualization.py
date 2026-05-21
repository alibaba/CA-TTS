import torch
from typing import Union, List, Tuple

# --- 新增的导入 ---
import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.cm as cm  # (新) 用于热力图颜色映射
import numpy as np          # (新) 用于处理张量以应用颜色映射

# ----------------------------------------------------------------------------
# 函数 1: (已修改，返回 noise)
# ----------------------------------------------------------------------------
def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]: # 修改了返回类型
    """
    Applies noise to the input image tensor (x_0) corresponding to a specific time step (t).
    
    ... (其余 docstring 未变) ...

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            1. The noisy image tensor (x_t).
            2. The noise tensor (epsilon) that was added.
    """
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t, noise): # (修改) 接受 noise 作为参数
        """Helper function to apply noise for a given step t"""
        # noise = torch.randn_like(x_0) # (修改) 噪声从外部传入
        
        # 确保 t 是一个整数索引
        t_int = int(t) if not isinstance(t, int) else t
        
        alphas_t = alphas_bar_sqrt[t_int]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t_int]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    
    # --- (修改) ---
    # 1. 在此处生成噪声
    noise_tensor = torch.randn_like(noisy_image)
    # 2. 将噪声传入 q_x
    image_tensor_cd = q_x(noisy_image, noise_step, noise_tensor) 
    # 3. 返回 noisy_image 和 noise
    return image_tensor_cd, noise_tensor
    # --- (修改结束) ---

# ----------------------------------------------------------------------------
# 函数 2: (已修改，收集 noise)
# ----------------------------------------------------------------------------
def prepare_noisy_samples(image_tensor: torch.Tensor, num_samples: int) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]: # 修改了返回类型
    """
    Generates a list of noisy image tensors...
    ... (其余 docstring 未变) ...

    Returns:
        Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
            1. noisy_samples: A list of image tensors (size num_samples + 1).
            2. all_t_steps:   A corresponding list of noise steps (t).
            3. noise_list:    A corresponding list of noise maps (epsilon) used.
                              (t=0 对应的 noise_map 为 0)
    """
    # DDPM parameters
    num_steps = 1000
    max_step_index = num_steps - 1  # 999

    # 1. 确定 *带噪* 样本的噪声步长索引
    noise_steps: List[int] = [] 
    
    for i in range(1, num_samples + 1):
        step_value = i * (max_step_index / num_samples)
        t_step = int(torch.round(torch.tensor(step_value)).item())
        t_step = max(1, min(max_step_index, t_step))
        noise_steps.append(t_step)
        
    all_t_steps: List[int] = [0] + noise_steps
    print(f"Calculated all noise steps (t): {all_t_steps}")
    
    # 2. 生成带噪图像列表
    
    # 列表第一个元素为 image_tensor 本身 (t=0)
    noisy_samples: List[torch.Tensor] = [image_tensor]
    
    # (修改) 创建一个列表来存储噪声图
    # t=0 时，噪声为0
    noise_list: List[torch.Tensor] = [torch.zeros_like(image_tensor)]
    
    # 遍历 t > 0 的步长来生成带噪图像
    for t_step in noise_steps:
        # (修改) 接收 noisy_image 和 added_noise
        noisy_image, added_noise = add_diffusion_noise(image_tensor.clone(), t_step)
        noisy_samples.append(noisy_image)
        noise_list.append(added_noise) # (修改) 存储噪声

    return noisy_samples, all_t_steps, noise_list # (修改) 返回 noise_list

# ----------------------------------------------------------------------------
# 主执行代码 (已修改)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- 1. 设置参数 ---
    
    # !!! 修改这里: 替换为您自己的图像路径 !!!
    INPUT_IMAGE_PATH = "./data/example_image.png"  # Replace with your image path
    
    OUTPUT_DIR = "output_noisy_images"
    HEATMAP_DIR = "output_heatmaps" # (新) 热力图输出目录
    IMAGE_SIZE = (336, 336)
    num_samples_to_gen = 8

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True) # (新)
    
    # --- 2. 加载并预处理图像 ---
    
    preprocess_transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(f"--- 准备输入 ---")
    
    try:
        image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"错误: 找不到图像文件 '{INPUT_IMAGE_PATH}'")
        print("请更新 INPUT_IMAGE_PATH 变量为有效的图像路径。")
        exit()
        
    # 形状变为 [1, 3, 336, 336]
    input_tensor = preprocess_transform(image).unsqueeze(0)
    
    print(f"Input image loaded: {INPUT_IMAGE_PATH}")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Requested noisy samples (num_samples): {num_samples_to_gen}")
    
    # --- 3. 调用函数生成加噪图像列表 ---
    print("\n--- 开始生成 ---")
    # (修改) 接收 list_of_noise_maps
    list_of_noisy_images, list_of_t_steps, list_of_noise_maps = prepare_noisy_samples(
        input_tensor, 
        num_samples_to_gen
    )
    
    # --- 4. 打印输出结果 (与之前相同) ---
    print("\n--- 生成完毕 ---")
    print(f"Total images generated: {len(list_of_noisy_images)}")
    print(f"Corresponding t-steps: {list_of_t_steps}")
    print(f"Total noise maps captured: {len(list_of_noise_maps)}") # (新)

    print("\n--- 图像统计 (均值/标准差) ---")
    for i in range(len(list_of_noisy_images)):
        t_step = list_of_t_steps[i]
        image_tensor = list_of_noisy_images[i]
        noise_map = list_of_noise_maps[i]
        print(f"  Img {i} (t={t_step:<3}): Mean={image_tensor.mean():.4f}, Std={image_tensor.std():.4f} | (Noise Mean={noise_map.mean():.4f}, Noise Std={noise_map.std():.4f})")

    # --- 5. 保存 *加噪后* 的图像 (与之前相同) ---
    print(f"\n--- 保存加噪图像到 '{OUTPUT_DIR}' ---")
    
    save_transform = T.ToPILImage()
    
    for i in range(len(list_of_noisy_images)):
        t_step = list_of_t_steps[i]
        noisy_tensor_bchw = list_of_noisy_images[i] # [1, C, H, W]
        
        noisy_tensor_chw = noisy_tensor_bchw.squeeze(0) # [C, H, W]
        img_tensor_0_1 = (noisy_tensor_chw * 0.5) + 0.5
        img_tensor_clamped = torch.clamp(img_tensor_0_1, 0.0, 1.0)
        pil_img = save_transform(img_tensor_clamped)
        
        output_filename = os.path.join(OUTPUT_DIR, f"noisy_image_t_{t_step:03d}.png")
        pil_img.save(output_filename)
        
    print(f"所有 {len(list_of_noisy_images)} 张加噪图像已保存。")

    # --- 6. (新) 保存 *热力图* 图像 ---
    print(f"\n--- 保存热力图到 '{HEATMAP_DIR}' ---")

    # 准备用于混合的 *原始* PIL 图像 (反标准化)
    base_img_tensor_0_1 = (input_tensor.squeeze(0) * 0.5) + 0.5
    base_pil_image = save_transform(base_img_tensor_0_1).convert('RGB')

    for i in range(len(list_of_noisy_images)):
        t_step = list_of_t_steps[i]
        noise_tensor = list_of_noise_maps[i] # [1, C, H, W]
        
        output_filename = os.path.join(HEATMAP_DIR, f"heatmap_blend_t_{t_step:03d}.png")

        if t_step == 0:
            # 对于 t=0，没有噪声，直接保存原图
            base_pil_image.save(output_filename)
            continue

        # --- 生成热力图 ---
        # 1. 计算噪声幅度。我们取 R,G,B 通道上噪声的平均绝对值
        # [1, C, H, W] -> [C, H, W] -> [H, W]
        noise_magnitude = noise_tensor.squeeze(0).abs().mean(dim=0)
        
        # 2. 将幅度归一化到 [0, 1] 范围，以便应用 colormap
        mag_min = noise_magnitude.min()
        mag_max = noise_magnitude.max()
        if mag_max == mag_min:
            # 避免除以零（虽然对于噪声基本不可能）
            mag_norm = torch.zeros_like(noise_magnitude)
        else:
            mag_norm = (noise_magnitude - mag_min) / (mag_max - mag_min)
            
        # 3. 转换为 NumPy 并应用 'jet' colormap
        # cm.jet() 返回一个 [H, W, 4] 的 RGBA 数组
        mag_np = mag_norm.cpu().numpy()
        heatmap_rgba_np = cm.jet(mag_np)
        
        # 4. 转换为 PIL Image
        # 将 [0, 1] 的浮点数转为 [0, 255] 的 uint8
        heatmap_pil = Image.fromarray((heatmap_rgba_np * 255).astype('uint8'), 'RGBA').convert('RGB')
        
        # 5. 将热力图与原图混合
        # alpha=0.5 表示 50% 原图 + 50% 热力图
        blended_image = Image.blend(base_pil_image, heatmap_pil, alpha=0.5)
        
        # 6. 保存
        blended_image.save(output_filename)

    print(f"所有 {len(list_of_noisy_images)} 张热力图已保存。")