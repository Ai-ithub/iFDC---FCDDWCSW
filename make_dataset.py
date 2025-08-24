import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import shutil

def add_well_log_features(img):
    height, width, _ = img.shape
    
    # اضافه کردن خط عمودی نشانگر عمق در سمت چپ
    img[:, 0:3, :] = (0.1, 0.1, 0.1)
    
    # اضافه کردن نشانگرهای عمق
    for i in range(0, height, height//10):
        img[i:i+2, 0:10, :] = (0.1, 0.1, 0.1)
        
    # اضافه کردن خطوط افقی نازک برای نشان دادن مقیاس
    for i in range(0, height, height//20):
        img[i:i+1, :, :] *= 0.8
    
    return img

def generate_synthetic_borehole_image(height=256, width=128, num_layers=10, add_features=True):
    img = np.zeros((height, width, 3))
    layer_height = height // num_layers
    rock_colors = [
        (0.6, 0.4, 0.2),   # Shale
        (0.9, 0.8, 0.6),   # Limestone
        (0.7, 0.7, 0.7),   # Sandstone
        (0.4, 0.6, 0.2),   # Siltstone
        (0.2, 0.2, 0.2),   # Basalt
    ]
    
    # اضافه کردن ویژگی‌های بیشتر برای واقعی‌تر شدن تصاویر
    for i in range(num_layers):
        color = rock_colors[np.random.randint(0, len(rock_colors))]
        start = i * layer_height
        end = (i + 1) * layer_height
        
        # تغییر ضخامت لایه‌ها به صورت تصادفی
        if i < num_layers - 1 and np.random.random() > 0.7:
            thickness_variation = np.random.randint(-layer_height//4, layer_height//4)
            end += thickness_variation
            if end <= start:
                end = start + layer_height // 3
        
        img[start:end, :, :] = color
        
        # اضافه کردن بافت به لایه‌ها
        texture = np.random.normal(0, 0.05, (end-start, width, 3))
        img[start:end, :, :] += texture
        
        # اضافه کردن ترک‌ها و شکستگی‌ها
        if np.random.random() > 0.8:
            crack_pos = np.random.randint(0, width)
            crack_width = np.random.randint(1, 5)
            crack_length = np.random.randint(5, end-start)
            crack_start = np.random.randint(start, end-crack_length)
            img[crack_start:crack_start+crack_length, crack_pos:crack_pos+crack_width, :] *= 0.7
    
    # اضافه کردن نویز به کل تصویر
    img += np.random.normal(0, 0.02, img.shape)
    img = np.clip(img, 0, 1)
    
    if add_features:
        img = add_well_log_features(img)
    
    return img

# ساخت و ذخیره دیتاست با تنوع بیشتر
output_dir = "synthetic_borehole_dataset"
os.makedirs(output_dir, exist_ok=True)

for i in tqdm(range(1000)):
    # تنوع در تعداد لایه‌ها
    num_layers = np.random.randint(5, 15)
    
    # تنوع در ابعاد تصویر
    height = np.random.choice([256, 320, 384])
    width = np.random.choice([128, 160, 192])
    
    img = generate_synthetic_borehole_image(height=height, width=width, num_layers=num_layers)
    plt.imsave(f"{output_dir}/img_{i:04d}.png", img)
    
    # ذخیره متادیتا برای هر تصویر
    with open(f"{output_dir}/img_{i:04d}_meta.txt", "w") as f:
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")
        f.write(f"num_layers: {num_layers}\n")


# کپی تصاویر واقعی به دیتاست
real_images_dir = "path/to/real/well/logs"
if os.path.exists(real_images_dir):
    for i, fname in enumerate(os.listdir(real_images_dir)):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy(
                os.path.join(real_images_dir, fname),
                os.path.join(output_dir, f"real_{i:04d}.png")
            )
