import matplotlib.pyplot as plt
import numpy as np
#灰度化
import cv2
# img = cv2.imread('train_MT/train/1.jpg')  # 读取图片
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度变化
# cv2.imshow("~",img_gray)
# cv2.waitKey(0)
# output_path = "train_MT/train/1_1.jpg"
# img_read = cv2.imread("train_MT/train/1.jpg",0) # 灰度图
# #img_gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
#
# plt.hist(img_read.flatten(), np.arange(-0.5, 256, 1), color='g')
# retval, img_thresh = cv2.threshold(img_read, 10, 50, cv2.THRESH_BINARY)
# # Show the images
# #plt.figure(figsize=[18,5])
# plt.subplot(121)
# plt.imshow(img_thresh, cmap="gray")
# plt.show()
# plt.axis('off')
# plt.savefig(output_path, dpi=96, bbox_inches='tight', pad_inches=0)
# plt.close()

import math
import cv2
import numpy as np
def yuzhichuli():
    output_path = "train_MT/train/1_1.jpg"
    img = cv2.imread('train_MT/train/1.jpg')  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_gray = cv2.equalizeHist(img_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 应用CLAHE
    img_gray = clahe.apply(img_gray)
    print(img_gray.shape)
    cv2.imshow("ia",img_gray)
    # 使用函数cv2.threshold()对数组进行二值化阈值处理
    #img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
    t,rst = cv2.threshold(img_gray, 30, 255, cv2.THRESH_TOZERO)
    #img_out = img_gray + rst
    cv2.imshow("rst",rst)
    cv2.imwrite("train_MT/train/1_1.jpg",rst)
    cv2.imshow("gios",img_gray)
    cv2.waitKey(0)
    cv2.imwrite("train_MT/train/1_2.jpg",img_gray)
    print("img=\n", img_gray)

def otsu():
    import cv2
    img = cv2.imread("train_MT/train/2.jpg", 0)
    img = cv2.equalizeHist(img)
    t1, thd = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    t2, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("img", img)
    cv2.imshow("thd", thd)
    cv2.imshow("otsu", otsu)
    cv2.waitKey()
    cv2.destroyAllWindows()

def adaptive():
    import cv2
    img = cv2.imread("train_MT/train/1.jpg", 0)
    t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    athdMEAN = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
    athdGAUS = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
    cv2.imshow("img", img)
    cv2.imshow("thd", thd)
    cv2.imshow("athdMEAN", athdMEAN)
    cv2.imshow("athdGAUS", athdGAUS)
    cv2.waitKey()
    cv2.destroyAllWindows()
if __name__=="__main__":
    yuzhichuli()
    # img = cv2.imread('train_MT/train/1.jpg')
    #
    # # 将图像转换为灰度图像
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # # 创建CLAHE实例
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # # 应用CLAHE
    # clipped_img = clahe.apply(gray_img)
    # img_gray1 = cv2.equalizeHist(gray_img)
    # # 显示原始图像和CLAHE后的图像
    # cv2.imshow('Original Image', clipped_img)
    # cv2.imshow('CLAHE Image', img_gray1)
    #
    # # 等待按键，然后关闭所有窗口
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()