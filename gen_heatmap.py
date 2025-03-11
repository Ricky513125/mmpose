import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm

# 定义输入和输出路径
input_root = "/mnt/ssd2/lingyu/Tennis/vid_frames_224"
output_root = "/mnt/ssd2/lingyu/mmpose/tennis_poses"

# MMPose 关键点检测命令模板
mmpose_cmd = """
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    /mnt/ssd2/lingyu/mmpose/checkpoints/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    /mnt/ssd2/lingyu/mmpose/checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input {input_img} --draw-heatmap --out-file {output_img}
"""

# 遍历视频文件夹
for video_folder in sorted(os.listdir(input_root)):
    video_path = os.path.join(input_root, video_folder)
    output_video_path = os.path.join(output_root, video_folder)

    if not os.path.isdir(video_path):
        continue  # 跳过非文件夹

    # 创建对应的输出文件夹
    os.makedirs(output_video_path, exist_ok=True)

    print(f"Processing video: {video_folder}")

    # 读取该视频文件夹下的所有图片
    image_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])

    for img_name in tqdm(image_files, desc=f"Processing {video_folder}"):
        input_img_path = os.path.join(video_path, img_name)
        output_img_path = os.path.join(output_video_path, img_name.replace(".jpg", "_heatmap.jpg"))
        output_npy_path = os.path.join(output_video_path, img_name.replace(".jpg", ".npy"))

        # 运行 MMPose 关键点检测
        cmd = mmpose_cmd.format(input_img=input_img_path, output_img=output_img_path)
        subprocess.run(cmd, shell=True, check=True)

        # 读取生成的 heatmap 图像
        if os.path.exists(output_img_path):
            heatmap_img = cv2.imread(output_img_path, cv2.IMREAD_GRAYSCALE)
            heatmap_npy = np.array(heatmap_img, dtype=np.float32)

            # 保存为 .npy 格式
            np.save(output_npy_path, heatmap_npy)

            # 删除 heatmap 可视化图片（如果不想保留）
            os.remove(output_img_path)
    break