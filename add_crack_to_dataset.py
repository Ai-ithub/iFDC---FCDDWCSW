import os
import cv2
import numpy as np
import random

input_dir = 'synthetic_borehole_dataset'
output_dir = 'synthetic_borehole_with_crack'
os.makedirs(output_dir, exist_ok=True)

# درصد تصاویر دارای ترک
crack_ratio = 0.3

# لیست تصاویر PNG
image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    add_crack = random.random() < crack_ratio
    if add_crack:
        h, w, _ = img.shape
        # مختصات تصادفی برای شروع ترک
        x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
        # طول و زاویه ترک
        length = random.randint(int(0.2*w), int(0.5*w))
        angle = random.uniform(0, 2*np.pi)
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w-1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h-1))
        # رسم ترک (خط مشکی)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, img)