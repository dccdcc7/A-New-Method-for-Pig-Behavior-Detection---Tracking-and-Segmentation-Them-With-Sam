import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
import cv2
import os
import sys
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
'''
需要修改的路径 line127、173、151、436、437、443
'''

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# 改变sam_checkpoint，model_type，device为你想要的模型
sam_checkpoint = "checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
sam.to(device=device)

from classification.predict import Predict

color_list = np.array(
    [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
        [0.635, 0.078, 0.184],
        [0.300, 0.300, 0.300],
        [0.600, 0.600, 0.600],
        [1.000, 0.000, 0.000],
        [1.000, 0.500, 0.000],
        [0.749, 0.749, 0.000],
        [0.000, 1.000, 0.000],
        [0.000, 0.000, 1.000],
        [0.667, 0.000, 1.000],
        [0.333, 0.333, 0.000],
        [0.333, 0.667, 0.000],
        [0.333, 1.000, 0.000],
        [0.667, 0.333, 0.000],
        [0.667, 0.667, 0.000],
        [0.667, 1.000, 0.000],
        [1.000, 0.333, 0.000],
        [1.000, 0.667, 0.000],
        [1.000, 1.000, 0.000],
        [0.000, 0.333, 0.500],
        [0.000, 0.667, 0.500],
        [0.000, 1.000, 0.500],
        [0.333, 0.000, 0.500],
        [0.333, 0.333, 0.500],
        [0.333, 0.667, 0.500],
        [0.333, 1.000, 0.500],
        [0.667, 0.000, 0.500],
        [0.667, 0.333, 0.500],
        [0.667, 0.667, 0.500],
        [0.667, 1.000, 0.500],
        [1.000, 0.000, 0.500],
        [1.000, 0.333, 0.500],
        [1.000, 0.667, 0.500],
        [1.000, 1.000, 0.500],
        [0.000, 0.333, 1.000],
        [0.000, 0.667, 1.000],
        [0.000, 1.000, 1.000],
        [0.333, 0.000, 1.000],
        [0.333, 0.333, 1.000],
        [0.333, 0.667, 1.000],
        [0.333, 1.000, 1.000]
    ]
).astype(np.float32)
#ssh -p 24631 root@connect.westb.seetacloud.com
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


def show_box(frame,id,box,ax):
    print(id)
    print(frame)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    #data_coords = ax.transData.transform([(x0,y0)])
    #data_x, data_y = data_coords[0]
    #print("shiaasdaaddaddad")
    # 使用转换后的数据坐标添加文字
    img_path = './img11/img11/' + r'{}'.format(id)+'-'+r'{}'.format(int(int(frame)/100)+1) + r'/{}.jpg'.format(frame)
    print(img_path)
    id = "id: " + str(id)
    #class_output = Predict(img_path)
    class_output = 'stand'
    #ax.text(x0+w-30, y0-5, id, fontdict={'fontsize': 8, 'fontweight': 'light', 'color': 'yellow'})
    ax.text(x0, y0, id + ' ' + class_output, fontdict={'fontsize': 12, 'fontweight': 'light', 'color': 'red'})

def save_masks(input_boxes,box2,masks):
    for i, mask in enumerate(masks):
        id = box2[i][1]
        print("id = ", id)
        frame = box2[i][0]
        print("frame = ", int(frame))
        num = int(frame)
        plt.figure(figsize=(3, 3))
        mask_image = show_mask(mask.cpu().numpy(), plt.gca(), id * 2, fix_color=True)
        plt.gca().imshow(mask_image, alpha=0.5)
        xmin, ymin, width, height = input_boxes.cpu()[i][0].item(), input_boxes.cpu()[i][1].item(), \
                                    input_boxes.cpu()[i][2].item(), input_boxes.cpu()[i][3].item()
        print(xmin, ymin, width, height)
        # plt.xlim(275,640)  # [325, 625.4, 598.2, 889.45]
        # plt.ylim(950,580)
        # print("sdsdfadgsdfgfdgsdfgsdfgsdfgsdfgsdfgsdfgsdfgsdfx:",str((xmin + width)/2/1920))
        # print("sdsdfadgsdfgfdgsdfgsdfgsdfgsdfgsdfgsdfgsdfgsdfy:",str((ymin + height)/2/1080))
        plt.xlim(xmin - 50, width + 50)  # [325, 625.4, 598.2, 889.45]
        plt.ylim(height + 50, ymin - 50)
        #print(output_path)
        output_path1 = './img11/img11/'
        folder_path = f"{output_path1}{id}-{int(int(frame)/100)+1}/"
        os.makedirs(folder_path, exist_ok=True)
        mask_output_path = f"{folder_path}{frame}.jpg"
        txt_output_path = f"{output_path1}{id}-{int(int(frame)/100)+1}.txt"
        with open(txt_output_path, 'a') as file:
            # 写入数字1和2，然后添加一个空格
            #file.write(str(((xmin + width) / 2) / 1920))
            file.write(str(((xmin + width) / 2) / 2560))
            file.write(',')
            #file.write(str(((ymin + height) / 2) / 1080))
            file.write(str(((ymin + height) / 2) / 1440))
            file.write('\n')
            # 写入换行符，以便开始新的一行
        plt.axis('off')
        plt.savefig(mask_output_path, dpi=48, bbox_inches='tight', pad_inches=0)
        plt.close()

def draw_classes(frame,id, box, ax):
        print(frame)
        print(id)
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.imshow(image)
        output_path1 = './img11/img11/'
        folder_path = f"{output_path1}{id}/"
        os.makedirs(folder_path, exist_ok=True)
        mask_output_path = f"{folder_path}{frame}.jpg"
        print(mask_output_path)
        output_cls = Predict(mask_output_path)
        output_cls = str(output_cls)
        x0, y0 = box[0], box[1]
        # w, h = box[2] - box[0], box[3] - box[1]
        # print(output_cls)
        # plt.text(x0, y0, output_cls, fontdict={'fontsize': 8, 'fontweight': 'light', 'color': 'green'})
        # plt.axis('off')
        # plt.gca().savefig(image_path, dpi=64, bbox_inches='tight', pad_inches=0)
        # plt.close()
        ax.text(x0, y0, output_cls, fontdict={'fontsize': 10, 'fontweight': 'light', 'color': 'green'})


def show_mask(mask, ax, id , fix_color=False):
    if fix_color:
        color = np.array([0,0,0,0.8])
    else:
        color = np.array([color_list[id][0],color_list[id][1],color_list[id][2],0.8])  #这里可以写一个np.array的颜色表，然后使用颜色表来实现这个功能，id直接变成索引，这样就不会每次颜色都不一样了
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


def process_image(image_path, output_path, dpi=48):
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

def process_image_by_multianchor(box2,box1,image_path, output_path, dpi=48):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    for item in box1:
        print(item)
        item[2] = item[0]+item[2]
        item[3] = item[1]+item[3]
    input_boxes = torch.tensor(box1, device=predictor.device)
    print(input_boxes)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    j=0
    for mask in masks:
        mask_image = show_mask(mask.cpu().numpy(), plt.gca(), j, fix_color=False)
        plt.gca().imshow(mask_image, alpha=0.5)
        j=j+1
    jj=0
    save_masks(input_boxes, box2, masks)
    for j,box in enumerate(input_boxes):
        id = box2[jj][1]
        frame = box2[jj][0]
        print(id)
        show_box(frame,id,box.cpu().numpy(), plt.gca())
        jj=jj+1
    plt.axis('off')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    #plt.show()
    #show_points(input_point, input_label, plt.gca())
    print(input_boxes.cpu()[0][0].item())
    # for i, mask in enumerate(masks):
    #     id = box2[i][1]
    #     print("id = ",id)
    #     frame = box2[i][0]
    #     print("frame = ",frame)
    #     plt.figure(figsize=(3, 3))
    #     mask_image = show_mask(mask.cpu().numpy(), plt.gca(), id*2,fix_color=True)
    #     plt.gca().imshow(mask_image, alpha=0.5)
    #     xmin, ymin, width, height = input_boxes.cpu()[i][0].item(),input_boxes.cpu()[i][1].item(),input_boxes.cpu()[i][2].item(),input_boxes.cpu()[i][3].item()
    #     print(xmin,ymin,width,height)
    #     # plt.xlim(275,640)  # [325, 625.4, 598.2, 889.45]
    #     # plt.ylim(950,580)
    #     plt.xlim(xmin-50,width+50)  # [325, 625.4, 598.2, 889.45]
    #     plt.ylim(height+50,ymin-50)
    #     print(output_path)
    #     output_path1 = './img17/img17/'
    #     folder_path = f"{output_path1}{id}/"
    #     os.makedirs(folder_path, exist_ok=True)
    #     mask_output_path = f"{folder_path}{frame}.png"
    #     txt_output_path = f"{folder_path}{id}.txt"
    #     with open(txt_output_path, 'a') as file:
    #         # 写入数字1和2，然后添加一个空格
    #         file.write(str((xmin+width/2)/1920))
    #         file.write(',')
    #         file.write(str((ymin-height/2)/1080))
    #         file.write('\n')
    #         # 写入换行符，以便开始新的一行
    #
    #     plt.axis('off')
    #     plt.savefig(mask_output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #
    # k=0
    # for k,box in enumerate(input_boxes):
    #     id = box2[k][1]
    #     frame = box2[k][0]
    #     output_path1 = './img17/img17/'
    #     image_path1 = f"{output_path1}{frame}.png"
    #     print("--------------",image_path1)
    #     image1 = cv2.imread(image_path1)
    #     image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image1)
    #     #print(id)
    #     draw_classes(frame,id,box.cpu().numpy(), plt.gca())
    #     k=k+1
    #     plt.axis('off')
    # #draw_classes(input_boxes, box2, masks)
    #     plt.savefig(image_path1, dpi=dpi, bbox_inches='tight', pad_inches=0)
    #     plt.show()

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
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.jpg")
            process_image_by_anchor(input_path, output_path)
            print("img{}".format(i), "done")

def getbox(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as source_file:
        # 读取所有行
        # 遍历数字1到9
        box_out1=[]
        box_out2=[]
        lines = source_file.readlines()
        for i in range(1, 4000):
            out_line = []
            num = str(i)
            for line in lines:
                if line.startswith(num+','):  # 检查行是否以特定数字和逗号开头
                    out_line.append(line)
                #print(out_line)
            if out_line:  # 如果找到匹配的行
                box = []
                box1 = []
                for item in out_line:
                    #print(item)
                    parts1 = item.split(",")[:2]
                    cleaned_data1 = ",".join(parts1)
                    number1 = [int(x) for x in cleaned_data1.split(",") if x]
                    #print(number1)
                    parts2 = item.split(",")[2:6]
                    cleaned_data2 = ",".join(parts2)
                    number2 = [float(x) for x in cleaned_data2.split(",") if x]
                    #print(number2)
                    box.append(number2)
                    box1.append(number1)
                box_out2.append(box1)
                box_out1.append(box)

        return box_out1,box_out2

def by_mulanchor():
    input_folder = "./img11/img1/"
    output_folder = "img11/img11"
    os.makedirs(output_folder, exist_ok=True)
    i = 1
    image_extension = ('.jpg')
    num=0
    for file_name in os.listdir(input_folder):
        if (file_name.lower().endswith(image_extension)):
            num+=1
    print(num)
    for j in range(num):
        # if (file_name.lower().endswith(image_extension)):
        gt_path = "./img11/gt/gt.txt"
        box1, box2 = getbox(gt_path)
        print(box1[i - 1])
        print(box2[i - 1])
        input_path = os.path.join(input_folder, r'{}.jpg'.format(i))
        print(input_path)
        output_path = os.path.join(output_folder, f"{i}.jpg")
        # print(box1)
        # print(i)
        print(output_path)
        process_image_by_multianchor(box2[i-1],box1[i-1],input_path, output_path)
        print("img{}".format(i), "done")
        i+=1


def img2video(folder_path,output_video_path):
    import cv2
    import os

    # 设置文件夹路径和输出视频的路径
    # folder_path = 'path/to/your/image/folder'
    # output_video_path = 'path/to/your/output/video.avi'

    # 获取所有图片的路径
    images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # 按名称排序图片
    images.sort()
    print(images)
    # 读取第一张图片来获取视频的帧率和尺寸
    frame = cv2.imread(folder_path+"/1.jpg")
    #frame = cv2.imread(os.path.join(folder_path, images[0]))
    #frame_size = (1080,1920)
    frame_size = (frame.shape[1], frame.shape[0])
    print(frame_size)
    fps = 16 # 每秒帧数，根据需要调整

    # 创建视频写入对象
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, frame_size)
    for i in range(len(images)+1):
        path_to_image = folder_path + r"/{}.jpg".format(i+1)
        frame = cv2.imread(path_to_image)
        if frame is not None:
            out.write(frame)
    # 释放资源
    out.release()

    print(f'Video has been created: {output_video_path}')

if __name__ =="__main__":
    # gt_path = "./img7/gt/gt.txt"
    # box1, box2 = getbox(gt_path)
    # #input_boxes = torch.tensor(box1[0], device=predictor.device)
    # #print(input_boxes)
    # print(box1[0])
    # print(box2[0])
    by_mulanchor()
    # input_path = "img17/img17"
    # output_path = "img17/img17/img17.mp4"
    # img2video(input_path,output_path)
# plt.figure(figsize=(10, 10))
# image_path = f"{output_path1}{frame}.png"
# print(image_path)
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
# output_cls = Predict(f"{folder_path}{frame}.png")
# # data_coords = ax.transData.transform([(x0,y0)])
# # data_x, data_y = data_coords[0]
# # print("shiaasdaaddaddad")
# # 使用转换后的数据坐标添加文字
# plt.text(xmin, ymin, output_cls, fontdict={'fontsize': 8, 'fontweight': 'light', 'color': 'green'})
# plt.savefig(image_path, dpi=96, bbox_inches='tight', pad_inches=0)