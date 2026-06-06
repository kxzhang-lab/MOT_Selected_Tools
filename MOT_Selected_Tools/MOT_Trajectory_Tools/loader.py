# 读取图像、加载标注
from typing import Dict, Optional, List
from collections import defaultdict
from pathlib import Path
import re

def get_image_files(image_folder) -> Dict[int, Path]:
    """获取图像序列文件夹中的所有图像文件"""
    image_files = {}
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    
    for ext in extensions:
        for img_path in Path(image_folder).glob(ext):
            stem = img_path.stem
            frame_id = parse_frame_id(stem)
            if frame_id is not None:
                image_files[frame_id] = img_path
    
    if not image_files:
        raise ValueError(f"在 {image_folder} 中未找到任何图像文件")
    
    print(f"找到 {len(image_files)} 张图像，帧范围: {min(image_files.keys())} ~ {max(image_files.keys())}")
    return image_files


def parse_frame_id(filename: str) -> Optional[int]:
    """从文件名中解析帧号"""
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])
    return None


def load_annotations(annotation_path) -> Dict[int, List[Dict]]:
    """加载标注文件"""
    annotations_by_frame = defaultdict(list)
    if not annotation_path.endswith('.txt'):
        raise Exception(f"标注文件{annotation_path}不是.txt文件。")
    frame_index,id_index = detect_column_defs(annotation_path)  # 帧列和id列的判断
    
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 6:
                frame_id = int(parts[frame_index])
                target_id = int(parts[id_index])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                
                annotations_by_frame[target_id].append({
                    'frame': frame_id,
                    'bbox': [x, y, w, h]
                })
    
    print(f"加载标注完成，共 {len(annotations_by_frame)} 帧包含标注")
    return dict(annotations_by_frame)


def detect_column_defs(anno_path):
    """
    根据数据内容自动判断标注格式是 (frame, id, ...) 还是 (id, frame, ...)
    通过统计第一列的变化次数来判断。
    """
    # 1. 取前100行作为样本，如果不足则取全部
    with open(anno_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    sample_lines = lines[:min(100, len(lines))]
    first_col_values = []
    
    for line in sample_lines:
        parts = line.split(',')
        if len(parts) >= 1:
            try:
                first_col_values.append(int(parts[0].strip()))
            except ValueError:
                # 如果转换失败，可能是其他格式，返回默认
                print("警告：无法解析第一列为整数，默认使用 (frame, id, ...)")
                return 0,1

    if len(first_col_values) < 2:
        print(f"警告：标注文件{anno_path}样本数量不足，无法判断frame，id的顺序")
        return None
    
    # 特殊情况：026视频的第一列全是-1，个人理解为该视频只有一个ID。因此第一列是ID，第二列是frame。
    if all(val == -1 for val in first_col_values):
        # print(f"视频 {video_name} 的第一列全为-1，判断为 (id, frame, ...)")
        return 1,0

    # 2. 统计第一列值发生变化的次数
    change_count = 0
    for i in range(1, len(first_col_values)):
        if first_col_values[i] != first_col_values[i-1]:
            change_count += 1

    # 3. 计算变化比例
    change_ratio = change_count / (len(first_col_values) - 1)
    
    # 4. 设置一个阈值，例如 0.5
    #    - 如果变化比例低 (<0.5)，说明是 frame (换帧时才变)
    #    - 如果变化比例高 (>=0.5)，说明是 id (变化频繁)
    threshold = 0.5
    if change_ratio < threshold:
        #print(f"视频 {video_name} 的第一列变化比例为 {change_ratio:.2f}，判断为 (frame, id, ...)")
        return 0,1
    else:
        #print(f"视频 {video_name} 的第一列变化比例为 {change_ratio:.2f}，判断为 (id, frame, ...)")
        return 1,0