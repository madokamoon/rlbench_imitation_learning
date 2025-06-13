import os
import glob
import cv2
import argparse
import re

def extract_timestamp(filename):
    """从文件名中提取时间戳"""
    # 移除扩展名并获取基本文件名
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # 尝试直接将整个文件名转换为浮点数(适用于纯数字时间戳)
    try:
        return float(name_without_ext)
    except ValueError:
        # 如果不成功，尝试提取所有数字
        try:
            numbers = re.findall(r'\d+\.\d+|\d+', name_without_ext)
            if numbers:
                # 将找到的第一个数字作为时间戳
                return float(numbers[0])
        except:
            pass
    
    # 如果无法提取时间戳，返回文件的修改时间
    return os.path.getmtime(filename)

def png_to_mp4(folder_path, output_filename="output.mp4", fps=30, step=10):
    """将指定文件夹中的所有PNG图片转换为MP4视频，每step张保存一帧"""
    try:
        # 确保文件夹路径存在
        if not os.path.exists(folder_path):
            print(f"错误: 文件夹 {folder_path} 不存在")
            return False

        # 获取所有PNG图片
        png_files = glob.glob(os.path.join(folder_path, "*.png"))
        
        if not png_files:
            print(f"错误: 在 {folder_path} 中没有找到PNG图片")
            return False

        # 按照文件名中的时间戳排序
        png_files.sort(key=extract_timestamp)
        
        # 读取第一张图片以获取尺寸
        first_image = cv2.imread(png_files[0])
        if first_image is None:
            print(f"错误: 无法读取图片 {png_files[0]}")
            return False
            
        height, width, layers = first_image.shape
        
        # 确保输出文件名有 .mp4 扩展名
        if not output_filename.lower().endswith('.mp4'):
            output_filename += '.mp4'
            
        # 定义视频输出路径
        output_path = os.path.join(folder_path, output_filename)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"错误: 无法创建视频写入器")
            return False
        
        # 逐帧添加图片到视频，每step张保存一次
        total_frames = len(png_files)
        selected_indices = range(0, total_frames, step)
        for idx, i in enumerate(selected_indices, 1):
            png_file = png_files[i]
            print(f"处理图片: {png_file} ({idx}/{len(selected_indices)})")
            image = cv2.imread(png_file)
            if image is not None:
                video_writer.write(image)
            else:
                print(f"警告: 无法读取图片 {png_file}，已跳过")
        
        # 释放视频写入器
        video_writer.release()
        
        print(f"视频已成功保存至: {output_path}")
        return True
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将PNG图片序列转换为MP4视频')
    parser.add_argument('folder', help='包含PNG图片的文件夹路径')
    parser.add_argument('-o', '--output', default='output.mp4', help='输出视频的文件名 (默认: output.mp4)')
    parser.add_argument('-f', '--fps', type=int, default=30, help='视频的帧率 (默认: 30)')
    parser.add_argument('-s', '--step', type=int, default=2, help='每多少张图片保存一帧 (默认: 10)')
    args = parser.parse_args()
    
    png_to_mp4(args.folder, args.output, args.fps, args.step)