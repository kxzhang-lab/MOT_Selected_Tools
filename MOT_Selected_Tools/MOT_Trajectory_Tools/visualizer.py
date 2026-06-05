# 绘制bbox、生成拼接图、生成视频
import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List
from MOT_Trajectory_Tools.extractor import *
from MOT_Trajectory_Tools.config import FRAME_OFFSET,SAMPLE_INTERVAL,FPS


def generate_visualization_key_images(start_frame: int, end_frame: int, add_key_frames: str, target_id:int,
                                      image_files: Dict[int, Path], annotations: Dict[int, List[Dict]], 
                                      output_dir: str):
    """
    生成关键帧的bbox可视化

    Params:
        start_frame: 起始帧（标注帧号）
        end_frame: 结束帧（标注帧号）
        add_key_frames: 人为指定的关键帧
        target_id: 指定轨迹的ID
        image_files: 所有图像序列
        annotations: 以帧为键组织的标注字典
        output_dir: 输出路径
    """
    key_frames = generate_key_frames(add_key_frames,start_frame,end_frame)
    h, w = get_video_resolution(image_files, start_frame)
    for anno_frame in tqdm(key_frames,desc='生成关键帧序列可视化结果'):
        frame, bbox = get_frame_box(anno_frame, h, w, image_files, annotations, target_id)
        vis_frame = visualize_bbox(bbox, frame, target_id, anno_frame)
        output_path = os.path.join(output_dir,f"frame_{(anno_frame-FRAME_OFFSET):06d}.jpg")
        cv2.imwrite(output_path, vis_frame)
    print("关键帧序列可视化均保存在{}文件夹内。".format(output_dir))


def get_video_resolution(image_files: Dict[int, Path], start_frame: int):
    """通过第一帧图像获取视频尺寸
    Params:
        frame_id: 待可视化的帧序号
        image_files: 所有图像序列
    """
    first_valid_frame = start_frame
    while first_valid_frame not in image_files:
        first_valid_frame += 1
    
    first_img = cv2.imread(str(image_files[first_valid_frame]))
    h, w = first_img.shape[:2]
    return h, w


def get_frame_box(frame_id:int, h:int, w:int, image_files: Dict[int, Path], 
                  annotations: Dict[int, List[Dict]],target_id:int):
    """获取帧图像和指定ID的box
    Params:
        frame_id: 待可视化的帧序号
        h: 视频的高度
        w: 视频的宽度
        image_files: 所有图像序列
        annotations: 以帧为键组织的标注字典
        target_id: 指定轨迹的ID号
    """
    img_frame_id = frame_id - FRAME_OFFSET
    if img_frame_id in image_files:
        frame = cv2.imread(str(image_files[img_frame_id]))
    else:                                           
        # 创建黑帧
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {frame_id} (MISSING)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # 获取当前帧的bbox
    bbox = None
    if frame_id in annotations:
        for ann in annotations[frame_id]:
            if ann['id'] == target_id:
                bbox = [int(v) for v in ann['bbox']]
                break
    return frame, bbox


def visualize_bbox(bbox: List[int], original_frame: np.ndarray,
                   target_id:int, frame_id:int):
    """在原图上画bbox
    Params:
        bbox: 物体边界框
        original_frame: 相应帧图像
        target_id: 目标轨迹序号
        frame_id: 标注帧序号
    """
    frame = original_frame.copy()
        
    if bbox is not None:
        x, y, bw, bh = bbox
        # 绘制细bbox（线宽1，红色）
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 0, 255), 1)
        # 在bbox上方标注ID
        cv2.putText(frame, f"ID:{target_id}", (x, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    # 添加帧号
    cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame
    

def generate_visualization_video(start_frame: int, end_frame: int, 
                                 image_files: Dict[int, Path], 
                                 annotations: Dict[int, List[Dict]],
                                 video_dir: str,target_id:int,
                                 output_name: str = "visualization.mp4") -> str:
    """
    生成左右拼接的可视化视频
    
    Args:
        start_frame: 起始帧（标注帧号）
        end_frame: 结束帧（标注帧号）
        image_files: 所有图像序列
        annotations: 以帧为键组织的标注字典
        video_dir: 视频保存文件夹路径
        target_id: 指定轨迹ID
        output_name: 输出视频文件名
        fps: 视频帧率
        sample_interval: 采样间隔
        
    Returns:
        输出视频路径
    """
    h,w = get_video_resolution(image_files, start_frame) # 获取原图分辨率
    out_w, out_h = w * 2 + 5, h  # 输出视频尺寸（拼接后宽度 = 原图宽 + 5px分隔线 + 原图宽）
    
    output_path = os.path.join(video_dir , output_name)  # 构造写视频的类
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (out_w, out_h))

    segments = get_target_timeline(start_frame, end_frame, target_id, annotations)  # 获取目标出现片段
    
    print(f"生成可视化视频: {output_path}")
    print(f"目标ID: {target_id}, 帧范围: {start_frame} ~ {end_frame}")
    print(f"采样间隔: {SAMPLE_INTERVAL}, 实际输出帧数: ~{(end_frame-start_frame+1)//SAMPLE_INTERVAL}")
    print(f"出现片段: {segments}")

    for frame_id in tqdm(range(start_frame, end_frame + 1, SAMPLE_INTERVAL), desc="生成视频"):
        frame,bbox = get_frame_box(frame_id, h, w, image_files, annotations, target_id)  # 获取图像和标注
        side_by_side = create_side_by_side_frame(frame_id, target_id, bbox, frame) # 创建拼接帧
        out_writer.write(side_by_side)
    
    out_writer.release()
    print(f"可视化视频已保存: {output_path}")
    return str(output_path)


def create_side_by_side_frame(frame_id: int, target_id:int, bbox: Optional[List[int]], 
                             original_frame: np.ndarray) -> np.ndarray:
    """
    创建左右拼接帧（原图+放大效果图）
        
    Args:
        frame_id: 帧号
        target_id: 指定轨迹的ID
        bbox: [x, y, w, h] 或 None（目标不存在）
        original_frame: 原始图像
            
    Returns:
        拼接后的图像
    """
    h, w = original_frame.shape[:2]
        
    if bbox is not None:
        # 左侧：原图 + 细bbox
        left_frame = visualize_bbox(bbox, original_frame, target_id, frame_id)

        # 右侧：放大效果图
        right_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        x, y, bw, bh = bbox
        if bw > 0 and bh > 0:
            # 计算放大区域（带边界检查）
            margin = 20
            crop_x1 = max(0, x - margin)
            crop_y1 = max(0, y - margin)
            crop_x2 = min(w, x + bw + margin)
            crop_y2 = min(h, y + bh + margin)
            
            # 裁剪目标区域
            crop = original_frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if crop.size > 0:
                # 放大到右侧区域大小
                right_h, right_w = right_frame.shape[:2]
                crop_h, crop_w = crop.shape[:2]
                
                # 计算缩放比例，保持宽高比
                scale = min(right_w / crop_w, right_h / crop_h)
                new_w = int(crop_w * scale)
                new_h = int(crop_h * scale)
                
                if new_w > 0 and new_h > 0:
                    resized = cv2.resize(crop, (new_w, new_h))
                    
                    # 居中放置
                    x_offset = (right_w - new_w) // 2
                    y_offset = (right_h - new_h) // 2
                    right_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                    
                    # 添加说明文字
                    cv2.putText(right_frame, f"Zoom: target ID {target_id}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            # 目标不存在时显示提示
            cv2.putText(right_frame, "Target not present", (w//4, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
        
        # 添加分隔线
        separator = np.ones((h, 5, 3), dtype=np.uint8) * 255
        side_by_side = np.hstack([left_frame, separator, right_frame])
        return side_by_side
