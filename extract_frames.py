# --------------------------------------------------------
# This script converts videos to folders of images.
# You can specify the source and destination folder in command-line parameters
# --------------------------------------------------------
import argparse
import glob
import os

import cv2


def process_videos(src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    cnt_vid = 1
    for video_name in glob.glob(os.path.join(src_path, "*.mp4")):
        video_name = os.path.basename(video_name)
        print("Processing " + video_name + "..." + str(cnt_vid) + "/" + str(len(glob.glob(os.path.join(src_path, "*.mp4")))))
        video_dst_path = os.path.join(dst_path, video_name.split('.')[0])
        if not os.path.exists(video_dst_path):
            os.mkdir(video_dst_path)
        vidcap = cv2.VideoCapture(os.path.join(src_path, video_name))
        (cap, frame) = vidcap.read()
        cnt_frame = 1
        while (cap):
            cv2.imwrite(
                os.path.join(video_dst_path, str(cnt_frame) + ".jpg"),
                frame)
            (cap, frame) = vidcap.read()
            cnt_frame += 1
        vidcap.release()
        cnt_vid += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', help='location of all videos you wish to process',
                        default="./img11", type=str)
    parser.add_argument('--dst_path', help='where to store frames',
                        default="./img11/img1", type=str)
    args = parser.parse_args()
    process_videos(args.src_path, args.dst_path)
