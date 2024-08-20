import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

sam_checkpoint = "checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam = sam_model_registry["vit_h"](checkpoint='./sam_vit_h_4b8939.pth')
# # sam.to(device="cuda")  # gpu
predictor = SamPredictor(sam)


def show_mask(mask, ax, random_color=False):
    # 掩膜部分
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def on_mouse(event, x, y, flags, param):
    global img, point1
    img3 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        predictor.set_image(img3)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
            )

        plt.figure(figsize=(10, 10))
        # 保存
        plt.imshow(img3)
        show_mask(masks, plt.gca())
        plt.savefig("1.png",dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    path = "notebooks/images/dog.jpg"
    img = cv2.imread(path)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

