# 提取目标轨迹
import numpy as np
from typing import Dict, List, Tuple
from MOT_Trajectory_Tools.config import KEY_FRAME_NUMS

def get_target_timeline(start_frame: int, end_frame: int, target_id:int,
                         annotations: Dict[int, Dict[int, List[float]]]) -> List[Tuple[int, int]]:
    """获取目标在指定时间范围内出现的连续片段
    Params:
        start_frame: 起始帧（标注帧号）
        end_frame: 结束帧（标注帧号）
        target_id: 指定轨迹ID
        annotations: 以帧为键组织的标注字典
    """
    segments = []
    current_segment = None
    
    tar_ann = annotations[target_id]
    for frame in range(start_frame, end_frame + 1):
        has_target = False
        if frame in tar_ann:
            has_target = True
        
        if has_target:
            if current_segment is None:
                current_segment = [frame, frame]
            else:
                current_segment[1] = frame
        else:
            if current_segment is not None:
                segments.append(tuple(current_segment))
                current_segment = None
    
    if current_segment is not None:
        segments.append(tuple(current_segment))
    
    return segments


def extract_bbox_annotations(start_frame: int, end_frame: int, annotations: Dict[int, Dict[int, List[float]]],
                             target_id: int, sample_interval: int = 1) -> Dict[int, List[int]]:
    """
    提取目标在指定范围内的bbox标注（原始坐标）
    
    Args:
        start_frame: 起始帧（标注帧号）
        end_frame: 结束帧（标注帧号）
        annotations: 以轨迹ID为key的标注字典
        target_id: 目标轨迹ID
        sample_interval: 采样间隔，1表示每帧都提取
        
    Returns:
        {frame_id: [x, y, w, h]}
    """
    bbox_dict = {}
    if target_id in annotations:
        for frame, bbox in annotations[target_id].items():
            is_save = start_frame <= frame <= end_frame and frame % sample_interval == 0
            if is_save: bbox_dict[frame] = [int(v) for v in bbox]
    return bbox_dict


def generate_key_frames(add_key_frames:str,start_frame:int,end_frame:int):
    """输出关键帧列表
    Params:
        add_key_frames: 人为指定的新增关键帧
        start_frame: 起始帧
        end_frame: 结束帧
    """
    added_key_frames = [int(x.strip()) for x in add_key_frames.split(',')]  # 人为新增关键帧
    # 生成关键帧列表（等间隔采样 + 用户指定）
    sampled_key_frames = list(np.linspace(start_frame, end_frame, KEY_FRAME_NUMS, dtype=int))
    if added_key_frames is not None:
        key_frames = list(set(added_key_frames) | set(map(int,sampled_key_frames)))  # 将新增关键帧和采样关键帧合并
        key_frames.sort()
    else:
        key_frames = sampled_key_frames  # 仅有采样关键帧
    return key_frames