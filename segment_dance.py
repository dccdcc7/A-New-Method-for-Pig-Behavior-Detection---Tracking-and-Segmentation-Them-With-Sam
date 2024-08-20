import cv2
import numpy as np
import glob as gb
import os
import imageio
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_dance_test import segment

COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

def match_obj(obj_list, obj_id):
    try:
        index = obj_list.index(obj_id)
    except:
        index = -1
    return index

def show_mask(i,mask, ax, random_color=False):
    # 掩膜部分
    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    color = np.array([15*i / 255, 5*i / 255, 10*i / 255, 0.6])
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

def show_points(coords, labels, ax, marker_size=5):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=3)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=3)

sam_checkpoint = "checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam = sam_model_registry["vit_h"](checkpoint='./sam_vit_h_4b8939.pth')
# # sam.to(device="cuda")  # gpu
predictor = SamPredictor(sam)


def txt2video(i,img):
    # img = cv2.imread("../dancetrack0026/test/" + "{}.jpg".format(i))
    # img2 = img

    with open('dancetrack0026.txt', 'r') as f:
        lines = f.readlines()
        object_list = []
        center_list = []
        img_list = []
        masks_list = []
        for line in lines:
            img_id = line.split(',')[0]
            if img_id == str(i):
                object_id = line.split(',')[1]
                object_list.append(object_id)
                x, y, w, h = int(float(line.split(',')[2])), int(float(line.split(',')[3])), int(float(line.split(',')[4])), int(
                    float(line.split(',')[5]))
                center1 = (int(int(x) + int(w) / 2), int(int(y) + int(h) / 2))
                center_list.append(center1)
                color = [int(c) for c in COLORS[int(object_id) % len(COLORS)]]
            # if img_id == str(int(i) + 1):
            #     object_id = line.split(',')[1]
            #     index = match_obj(object_list, object_id)
            #     x, y, w, h = int(float(line.split(',')[2])), int(float(line.split(',')[3])), int(float(line.split(',')[4])), int(
            #         float(line.split(',')[5]))
            #     center2 = (int(int(x) + int(w) / 2), int(int(y) + int(h) / 2))
            #     color = [int(c) for c in COLORS[int(object_id) % len(COLORS)]]
            #     if index != -1:
                img2 = cv2.rectangle(img, (x, y), (x + w, y + h), color)
                #img2 = cv2.arrowedLine(img2, center_list[1], center1, color, 1, 8, 0, 0.5)
                img2 = cv2.putText(img2, 'ID-' + str(int(object_id)), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                img_list.append(img2)
                    # print(len(center_list))
                    # for i in range(len(center_list)):
                    #     input_point1 = np.array([[a[i][0], a[i][1]]])
                    #     input_label1 = np.array([i])
                    #     # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
                    #     predictor.set_image(img2)
                    #     # show_points(input_point1, input_label1, plt.gca())
                    #     masks1, scores, logits = predictor.predict(
                    #         point_coords=input_point1,
                    #         point_labels=input_label1,
                    #         multimask_output=False,
                    #     )
                    #     masks_list.append(masks1)
                    #     plt.figure(figsize=(10, 10))
                    #     plt.imshow(img2)
                    #     for i in range(len(a)):
                    #         # 保存
                    #         show_points(np.array([[a[i][0], a[i][1]]]), np.array([i]), plt.gca())
                    #         show_mask(i, masks_list[i], plt.gca())
                    #     plt.savefig("afdlgho.png", dpi=500, bbox_inches='tight')
        # print(object_list)
        # print(center_list)
        #return object_list,center_list
        return center_list,img2

def mix_pre(img):
    centerlist,img2 = txt2video(1,img)
    segment(centerlist,img2)


if __name__ == '__main__':
    path = "1.jpg"
    img = cv2.imread(path)
    # img = cv2.imread(path)
    # centerlist,img2 = txt2video(1,img)
    # segment(centerlist,img2)
    mix_pre(img)
    # print(a)
    # print(b)
    #cv2.imwrite("ngosui.jpg",img2)
    # print(a[0])
    # print(b[0][0])
    # print(b[0][1])
    # print(len(b))
    #img2 =  cv2.putText(img, 'ID-' + str(int(1)), (b[0][0],b[0][1]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
    #img_out1 = on_mouse(img)

    #img_out = on_mouse(img_out1, 365,300)
    #cv2.imwrite("true_out",img_out)




