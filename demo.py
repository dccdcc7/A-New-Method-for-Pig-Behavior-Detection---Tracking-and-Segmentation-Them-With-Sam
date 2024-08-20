#这是一个seganythng的demo
#先导入必要的包
import os
import sys
import time
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


#这里是导入segment_anything包
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry


#输入必要的参数
model_path = 'checkpoint/sam_vit_b_01ec64.pth'
image_path = '1.png'
output_folder = "output"
#output_path = r'C:\AI\segment-anything-main\demo_mask\demo.png'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)
#这里的model_path是模型的路径，image_path是输入的图片路径，output_path是输出的图片路径

#这里是加载模型



#官方demo加载模型的方式
sam = sam_model_registry["vit_b"](checkpoint=model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam.to(device)

#输出模型加载完成的current时间

current_time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Model loaded done", current_time1)
print("1111111111111111111111111111111111111111111111111111")

#这里是加载模型，这里的model_path是模型的路径，sam_model_registry是模型的名称

#这里是加载图片
image = cv2.imread(image_path)
#输出图片加载完成的current时间
current_time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Image loaded done", current_time2)
print("2222222222222222222222222222222222222222222222222222")

#这里是加载图片，这里的image_path是图片的路径

#这里是预测,不用提示词,进行全图分割
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(image)

#使用提示词,进行局部分割
predictor = SamPredictor(sam)
predictor.set_image(image)
# masks, _, _ = predictor.predict(point_coords=np.array([[356, 378]]), point_labels=np.array([1]))
input_point = np.array([[356, 378]])
input_label = np.array([1])
masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,multimask_output=True,)
print(type(masks))
print(masks.shape)

#输出预测完成的current时间
current_time3 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Predict done", current_time3)
print("3333333333333333333333333333333333333333333333333333")

# #展示预测结果img和mask
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.subplot(1, 2, 2)
# plt.imshow(masks[0])
# plt.show()

print(masks[0])


# # mask_array = masks[0]['segmentation']


# # mask_uint8 = (mask_array * 255).astype(np.uint8)

# cv2.imwrite(output_path, mask_uint8)

#循环保存mask
# 遍历 masks 列表并保存每个掩码
# height, width, _ = image.shape
#
# for i, mask in enumerate(masks):
#     mask_array = mask
# #    mask_array_reshaped = mask_array.reshape((height, width))
#     mask_uint8 = (mask_array * 255).astype(np.uint8)
#
#     # 为每个掩码生成一个唯一的文件名
#     output_file = os.path.join(output_folder, f"mask_{i+1}.png")
#
#     # 保存掩码
#     cv2.imwrite(output_file, mask_uint8)

#输出完整的mask
# 获取输入图像的尺寸
height, width, _ = image.shape

# 创建一个全零数组，用于合并掩码
merged_mask = np.zeros((height, width), dtype=np.uint8)
print(masks.shape)
# 遍历 masks 列表并合并每个掩码
for i, mask in enumerate(masks):
    mask_array = mask
    print("单通道",mask_array.shape)
#    mask_array_reshaped = mask_array.reshape((height, width))
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    print("空的掩码",type(mask_uint8))
    print("图片的格式",type(image))
    # 为每个掩码生成一个唯一的文件名
    output_file = os.path.join(output_folder, f"mask_{i+1}.png")

    # 保存掩码
    cv2.imwrite(output_file, mask_uint8)

    # 将当前掩码添加到合并掩码上
    merged_mask = np.max((merged_mask, mask_uint8), axis=0)


masks_merged = (masks * 255).astype(np.uint8)
#print(masks)
#cv2.imwrite("1231.jpg",masks)
masks_merged= masks_merged.astype(np.uint8)
image_uint8 = (image * 255).astype(np.uint8)
image_uint8 = np.transpose(image_uint8, (2, 0, 1))
print(image_uint8)
#merged_mask_image = np.max((masks, image_uint8), axis=0)
merged_mask_image = image_uint8 + masks_merged
# 保存合并后的掩码
merged_output_file = os.path.join(output_folder, "mask_all.png")
merged_output_file1 = os.path.join(output_folder, "mask_all1.png")
cv2.imwrite(merged_output_file, merged_mask)
cv2.imwrite(merged_output_file1, merged_mask_image)
# #释放cv2
# cv2.destroyAllWindows()
