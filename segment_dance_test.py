import cv2
import numpy as np
import glob as gb
import os
import imageio
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import random

random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
sam_checkpoint = "checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam = sam_model_registry["vit_h"](checkpoint='./sam_vit_h_4b8939.pth')
# # sam.to(device="cuda")  # gpu
predictor = SamPredictor(sam)


def show_points(coords, labels, ax, marker_size=5):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=3)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=3)

def show_mask(i,mask, ax, random_color=False):
    # 掩膜部分
    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    color = np.array([12*i / 255, 6*i / 255, 10*i / 255, 0.6])
    # random.seed(100)
    # #
    # # # 生成1到10之间的随机整数
    # random_integer = random.randint(1, 10)
    # color = [int(c) for c in COLORS[int(random_integer) % len(COLORS)]]
    # print(color)
    # color[0]/=255
    # color[1] /= 255
    # color[2] /= 255
    # color.append(0.6)
    print(i)
    print(((i+1)/12)*100,"%")
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color
    ax.imshow(mask_image)



def segment(a,img3):
    masks_list=[]
    print(len(a))
    for i in range(len(a)):
        input_point1 = np.array([[a[i][0], a[i][1]]])
        input_label1 = np.array([i])
        #img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        predictor.set_image(img3)
        #show_points(input_point1, input_label1, plt.gca())
        masks1, scores, logits = predictor.predict(
            point_coords=input_point1,
            point_labels=input_label1,
            multimask_output=False,
        )
        masks_list.append(masks1)
    # plt.figure(figsize=(10, 10))
    # # 保存
    # plt.imshow(img3)

    # input_point2 = np.array([[a[1][0], a[1][1]]])
    # input_label2 = np.array([1])
    # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    # predictor.set_image(img3)
    # masks2, scores, logits = predictor.predict(
    #     point_coords=input_point2,
    #     point_labels=input_label2,
    #     multimask_output=False,
    # )
    #
    # input_point3= np.array([[a[2][0], a[2][1]]])
    # input_label3 = np.array([2])
    # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    # predictor.set_image(img3)
    # masks3, scores, logits = predictor.predict(
    #     point_coords=input_point3,
    #     point_labels=input_label3,
    #     multimask_output=False,
    # )
    plt.figure(figsize=(10, 10))
    plt.imshow(img3)
    for i in range(len(a)):
        # 保存
        show_points(np.array([[a[i][0], a[i][1]]]), np.array([i]), plt.gca())
        show_mask(i,masks_list[i], plt.gca())
    plt.savefig("afdlgho.png", dpi=500, bbox_inches='tight')
    # show_mask(masks2, plt.gca())
    # show_mask(masks3, plt.gca())

if __name__ == "__main__":
    a = np.array(
        [(861, 479), (1201, 446), (528, 443), (972, 491), (1199, 482), (720, 393), (211, 441), (1019, 422), (644, 457),
         (365, 386), (477, 461), (767, 450)])
    img_path = "1.jpg"
    img = cv2.imread(img_path)
    segment(a,img)

