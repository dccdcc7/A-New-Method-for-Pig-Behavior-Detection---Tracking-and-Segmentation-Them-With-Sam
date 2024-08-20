import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# 改变sam_checkpoint，model_type，device为你想要的模型
sam_checkpoint = "checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
sam.to(device=device)

#裁剪画布上的某块区域
def save_cropped_image(image_path, output_path, crop_rect):
    """
    裁剪画布上的矩形区域并保存。

    参数:
    - image_path: 原始图像的路径。
    - output_path: 裁剪后的图像保存路径。
    - crop_rect: 裁剪区域的坐标，格式为 (xmin, ymin, width, height)。
    """
    # 读取图像
    image = mpimg.imread(image_path)

    # 创建画布和轴
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')  # 隐藏轴

    # 确定裁剪区域
    xmin, ymin, width, height = crop_rect
    xmax = xmin + width
    ymax = ymin + height

    # 裁剪画布
    bbox = [xmin / fig.bbox_inches.width, ymin / fig.bbox_inches.height,
            width / fig.bbox_inches.width, height / fig.bbox_inches.height]
    # 保存裁剪后的画布
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    # 关闭画布
    plt.close()

def cor_img(mask_img,box,output_path):
    xmin, ymin = box[0], box[1]
    width, height = box[2] - box[0], box[3] - box[1]
    fig, ax = plt.subplots()
    ax.imshow(mask_img)
    ax.axis('off')  # 隐藏轴
    bbox = [xmin / fig.bbox_inches.width, ymin / fig.bbox_inches.height,
            width / fig.bbox_inches.width, height / fig.bbox_inches.height]
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_mask(mask, ax, id,random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([id*20/255, id*20/255, 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    #
    # img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 1)))


def process_image(image_path, output_path, dpi=96):
    sam_checkpoint = "checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # model: Sam类型，这是传入的模型实例，通常是一个深度学习模型，用于生成图像的分割掩码。
    # points_per_side: int类型，可选，默认值为32。这个参数指定了在每个边界上生成的点的数量。在分割任务中，这些点用于定义对象的轮廓。
    # points_per_batch: int类型，指定了每个批次处理的点的数量。
    # pred_iou_thresh: float类型，默认值为0.88。这个阈值用于预测时的交并比（Intersection over Union, IoU），用于评估预测掩码与真实掩码的重叠程度。
    # stability_score_thresh: float类型，默认值为0.95。稳定性分数阈值，用于评估生成的掩码的稳定性。
    # stability_score_offset: float类型，默认值为1.0。这个偏移量与稳定性分数一起使用，可能用于调整稳定性评分的严格程度。
    # box_nms_thresh: float类型，默认值为0.7。这是用于边界框的非极大值抑制（Non-Maximum Suppression, NMS）的阈值，用于去除重叠的边界框。
    # crop_n_layers: int类型，指定了在进行图像裁剪时使用的层数。
    # crop_nms_thresh: float类型，用于图像裁剪后的NMS阈值。
    # crop_overlap_ratio: float类型，指定了裁剪区域之间的重叠比例。
    # crop_n_points_downscale_factor: int类型，用于在裁剪过程中调整点的数量的缩放因子。
    # point_grids: List[np.ndarray]类型，可选。这是一个包含点网格的列表，每个网格是一个NumPy数组。
    # min_mask_region_area: int类型，默认值为0。这是生成掩码的最小区域面积，小于这个值的掩码可能会被忽略。
    # output_mode: str类型，默认值为"binary_mask"。这指定了输出掩码的模式，例如二进制掩码或其他形式。
    mask_generator_2 = SamAutomaticMaskGenerator(model=sam, points_per_side=32, pred_iou_thresh=0.86,
                                                 stability_score_thresh=0.92, crop_n_layers=1,
                                                 crop_n_points_downscale_factor=2, min_mask_region_area=40)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask2 = mask_generator_2.generate(image)
    height,width, _ =image.shape
    figsize = (width/dpi,height/dpi)
    plt.figure(figsize=figsize,dpi=dpi)
    plt.imshow(image)
    show_anns(mask2)
    plt.axis('off')
    plt.savefig(output_path,dpi=dpi,bbox_inches='tight',pad_inches=0)
    plt.close()

def process_image_by_anchor(image_path, output_path, dpi=96):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    #input_point = np.array([[500, 375]])
    #input_label = np.array([1])
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    #show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    # plt.show()
    # plt.savefig('my_figure.png', dpi=96, bbox_inches='tight',)
    # masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
    #                                           multimask_output=False, )
    input_box = np.array([968.2,6.35,1152.5,129.7])
    masks,scores, logits= predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :],multimask_output=False,)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        #show_points(input_point, input_label, plt.gca())
        plt.savefig(output_path, dpi=64, bbox_inches='tight', pad_inches=0)
        plt.show()

def process_image_by_multianchor(image_path, output_path, dpi=96):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    input_boxes = torch.tensor([
        [325, 625.4, 598.2, 889.45],
    ], device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    j=0
    for mask in masks:
        mask_image = show_mask(mask.cpu().numpy(), plt.gca(), j, random_color=True)
        plt.gca().imshow(mask_image, alpha=0.5)
        j=j+1

    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    #plt.axis('off')
    #show_points(input_point, input_label, plt.gca())
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.show()
    print(input_boxes.cpu()[0][0].item())
    for i, mask in enumerate(masks):
        plt.figure(figsize=(10, 10))
        mask_image = show_mask(mask.cpu().numpy(), plt.gca(), i*2,random_color=False)
        plt.gca().imshow(mask_image, alpha=0.5)
        xmin, ymin, width, height = input_boxes.cpu()[i][0].item(),input_boxes.cpu()[i][1].item(),input_boxes.cpu()[i][2].item(),input_boxes.cpu()[i][3].item()
        plt.xlim(275,640)  # [325, 625.4, 598.2, 889.45]
        plt.ylim(950,580)
        plt.xlim(xmin-50,width+50)  # [325, 625.4, 598.2, 889.45]
        plt.ylim(height+50,ymin-50)
        print(output_path)
        output_path = output_path.rstrip('5_result.png')
        mask_output_path = f"{output_path}mask{i}.png"
        plt.axis('off')
        plt.savefig(mask_output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()


def all_set():
    input_folder = "notebooks/images"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # ssh -p 27894 root@connect.westb.seetacloud.com
    i = 0
    image_extension = ('.png', '.jpg', '.jpeg')
    for file_name in os.listdir(input_folder):
        i += 1
        if (file_name.lower().endswith(image_extension)):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_result.png")
            process_image(input_path, output_path)
            print("img{}".format(i), "done")

def by_anchor():
    input_folder = "notebooks/images"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    i = 0
    image_extension = ('.png', '.jpg', '.jpeg')
    for file_name in os.listdir(input_folder):
        i += 1
        if (file_name.lower().endswith(image_extension)):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_result.png")
            process_image_by_anchor(input_path, output_path)
            print("img{}".format(i), "done")

def by_mulanchor():
    input_folder = "notebooks/images"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    i = 0
    image_extension = ('.png', '.jpg', '.jpeg')
    for file_name in os.listdir(input_folder):
        i += 1
        if (file_name.lower().endswith(image_extension)):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_result.png")
            process_image_by_multianchor(input_path, output_path)
            print("img{}".format(i), "done")


if __name__ =="__main__":
    by_mulanchor()

