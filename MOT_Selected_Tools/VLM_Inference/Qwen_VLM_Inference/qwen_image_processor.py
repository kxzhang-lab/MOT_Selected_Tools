import os
import mimetypes
import base64
import numpy as np
from typing import List, Dict

def image_to_data_url(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as f:
        image_data = f.read()
    mime_type,_ = mimetypes.guess_type(image_path)

    if mime_type is None or not mime_type.startswith('image/'):
        mime_type = 'image/png'

    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded}"

def build_image_content(frames: List[str], sample_num: int) -> List[Dict]:
    """
    将图像序列构建为千问API的content格式
    
    Args:
        frames: 图像路径列表
        sample_num: 用于输出描述的图像数目
    Returns:
        content列表，每个元素是一个图像字典
    """
    content = []
    sample_num = sample_num if sample_num else len(frames)
    sample_index = np.linspace(0,len(frames)-1,sample_num,dtype=int)
    sample_index = list(map(lambda x:int(x), list(sample_index)))
    sampled_frames = [frames[index] for index in sample_index]  # 采样
    for frame_path in sampled_frames:
        if not os.path.exists(frame_path):
            print(f"警告: 图像不存在 - {frame_path}")
            continue
        data_url = image_to_data_url(frame_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        })
    return content