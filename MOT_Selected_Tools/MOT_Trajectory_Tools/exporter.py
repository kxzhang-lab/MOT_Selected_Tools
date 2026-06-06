# 保存关键帧、保存JSON、生成模型输入
import os
import json
from typing import Dict, Any, List
from MOT_Trajectory_Tools.extractor import *

def export_bbox_json(start_frame: int, end_frame: int, 
                     annotations: Dict[int, List[Dict]],
                     target_id: int, json_dir: str,
                     sample_interval: int = 1) -> Dict[str, Any]:
    """
    导出bbox标注为JSON文件
    
    Args:
        start_frame: 起始帧
        end_frame: 结束帧
        annotations: 以轨迹ID为键的字典
        target_id: 轨迹ID序号
        json_dir: .json文件的保存文件夹路径
        sample_interval: 采样间隔
        
    Returns:
        bbox字典
    """
    bbox_dict = extract_bbox_annotations(start_frame, end_frame, annotations, 
                                         target_id, sample_interval)
    
    output_data = {
        "target_id": target_id,
        "frame_range": [start_frame, end_frame],
        "sample_interval": sample_interval,
        "frame_offset_note": "Annotation frames start from 1, image frames start from 0",
        "bbox_format": "[x1, y1, w, h]",
        "annotations": bbox_dict
    }
    
    output_path = os.path.join(json_dir , f"target_{target_id}_bbox.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"bbox标注已保存: {output_path}")
    

def generate_videollama3_input(video_path: str, 
                                start_frame: int, end_frame: int, add_key_frames:str,
                                json_dir:str, target_id:int, max_frames: int = 180) -> Dict:
    """
    生成VideoLLAMA3的conversation输入
    
    Args:
        video_path: 可视化视频路径
        bbox_json_path: bbox标注JSON路径
        start_frame: 起始帧
        end_frame: 结束帧
        add_key_frames: 除了等间隔采样外，人为增加的关键帧
        json_dir: json文件夹路径
        target_id: 指定轨迹的ID
        max_frames: 最大帧数限制
        
    Returns:
        VideoLLAMA3 conversation格式的字典
    """
    # 加载bbox数据
    bbox_json_path = os.path.join(json_dir , f"target_{target_id}_bbox.json")
    with open(bbox_json_path, 'r') as f:
        bbox_data = json.load(f)
    
    bbox_dict = bbox_data["annotations"]
    
    # 生成关键帧列表（等间隔采样 + 用户指定）
    key_frames = generate_key_frames(add_key_frames,start_frame,end_frame)

    # 如果指定了关键帧，只保留关键帧的bbox信息，否则使用全部bbox
    if key_frames:
        filtered_bbox = {k: v for k, v in bbox_dict.items() if int(k) in key_frames}
        bbox_dict = filtered_bbox
    
    # 采样限制（如果bbox太多）
    if len(bbox_dict) > max_frames:
        # 等间隔采样
        frame_list = sorted(bbox_dict.keys())
        step = len(frame_list) // max_frames
        sampled_frames = frame_list[::step][:max_frames]
        bbox_dict = {k: bbox_dict[k] for k in sampled_frames}
    
    # 构建bbox信息文本
    bbox_info = f"""
        Target ID: {target_id}
        Frame range: {start_frame} ~ {end_frame}
        Bounding boxes (sampled): {len(bbox_dict)} frames
        {json.dumps(bbox_dict, indent=2)}
        BBox format: [x1, y1, w, h]
        The target corresponding to these bounding boxes should be described.
        Please focus only on this target.
        """
    
    annotation_rules = """
        Generate a language description for UAV-MOT grounding.
        Rules:
        1. Do not rely only on static color.
        2. Include spatial anchor or dynamic behavior.
        3. Describe observable visual evidence only.
        4. If behavior changes over time, describe the temporal transition.
        5. Output Chinese description, English description, and JSON fields.
        """
    
    # 添加关键帧信息（如果有）
    key_frame_info = ""
    if key_frames:
        key_frame_info = f"\nKey behavior frames: {key_frames}. Pay attention to behavior changes at these timestamps.\n"
    
    question = f"""
        {bbox_info}

        {annotation_rules}

        {key_frame_info}

        Please describe the target in the video.
        Output:
        1. Chinese description
        2. English description
        3. Structured JSON
        """
    
    conversation = {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": 1,  # 每帧采样
                            "max_frames": min(len(bbox_dict), max_frames)
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
    }
    
    # 保存到文件
    output_path = os.path.join(json_dir , f"target_{target_id}_videollama3_input.json")
    with open(output_path, 'w') as f:
        json.dump(conversation, f, indent=2)
    
    print(f"VideoLLAMA3输入已保存: {output_path}")
    return conversation