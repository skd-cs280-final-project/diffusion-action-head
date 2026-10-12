# 将指定路径下的图像合并为mp4视频，并保存到./videos/目录下
# 需要调整 image_folder 为你想要合成视频的图像文件夹路径
# 运行示例： python tools/visual/get_video.py


import cv2
import os
import re
from pathlib import Path

def natural_sort_key(file_name):
    """
    自然排序（数字按数值大小排序，而不是字符串排序）
    例如：img1, img2, img10 而不是 img1, img10, img2
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return alphanum_key(file_name)

def images_to_video(image_folder, output_video, fps=30, size=None):
    """
    将文件夹中的所有图像按名称顺序合成视频

    参数:
        image_folder: 图像文件夹路径
        output_video: 输出视频文件路径（如：output.mp4）
        fps: 帧率（每秒帧数），默认30
        size: 视频尺寸 (width, height)，如果为None则使用第一张图片的尺寸
    """
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []

    for file in os.listdir(image_folder):
        ext = Path(file).suffix.lower()
        if ext in image_extensions:
            image_files.append(file)

    if not image_files:
        print(f"错误：在文件夹 '{image_folder}' 中未找到图像文件")
        return False

    # 按名称自然排序
    image_files.sort(key=natural_sort_key)

    print(f"找到 {len(image_files)} 张图像")
    print(f"第一张: {image_files[0]}")
    print(f"最后一张: {image_files[-1]}")

    # 读取第一张图片获取尺寸
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)

    if first_image is None:
        print(f"错误：无法读取图像 '{first_image_path}'")
        return False

    height, width = first_image.shape[:2]

    # 如果指定了size，则使用指定尺寸
    if size is not None:
        width, height = size

    print(f"视频尺寸: {width} x {height}")
    print(f"帧率: {fps} FPS")

    # 创建视频写入对象
    # 使用 'mp4v' 编解码器（适用于MP4格式）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"错误：无法创建视频文件 '{output_video}'")
        return False

    # 逐张读取并写入视频
    processed_count = 0
    for file_name in image_files:
        image_path = os.path.join(image_folder, file_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"警告：跳过无法读取的图像 '{file_name}'")
            continue

        # 如果需要调整尺寸
        if size is not None or image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height))

        video_writer.write(image)
        processed_count += 1

        # 显示进度
        if processed_count % 10 == 0:
            print(f"进度: {processed_count}/{len(image_files)}")

    # 释放资源
    video_writer.release()

    print(f"✅ 视频合成完成！")
    print(f"处理了 {processed_count} 张图像")
    print(f"输出文件: {output_video}")

    return True


def main():
    """主函数 - 使用示例"""

    # ===== 配置参数 =====
    # image_folder = "/home/wangluo/AerialVLA/eval_results/aerial_vla/seen_valset/NYCEnvironmentMegapa/ecbcca4e-9259-48ae-99e2-67b31f873fb6/downcamera"          # 图像文件夹路径（请修改为你的路径）
    image_folder = "/home/wangluo/VLN/eval_results/aero_vla/unseen_object_valset/NewYorkCity/success_cf7418a9-d733-4eb8-9783-3ab0c7667c6e/frontcamera"          # 图像文件夹路径（请修改为你的路径）

    video_name = f"{image_folder.split('/')[-2]}_{image_folder.split('/')[-1]}"
    output_video = f"./videos/{video_name}.mp4"      # 输出视频路径
    fps = 10                           # 帧率
    size = None                        # 视频尺寸 (width, height)，None表示使用原图尺寸
    # size = (1920, 1080)              # 例如：全高清尺寸

    os.makedirs("./videos", exist_ok=True)  # 确保输出目录存在

    # ===== 执行合成 =====
    if not os.path.exists(image_folder):
        print(f"错误：文件夹 '{image_folder}' 不存在")
        return

    images_to_video(image_folder, output_video, fps, size)


if __name__ == "__main__":
    main()