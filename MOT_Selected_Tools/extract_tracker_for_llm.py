"""
航拍MOT数据集预处理脚本（VideoLLAMA3专用版）
功能：
1. 生成左右拼接视频（原图+放大图），细bbox显示目标
2. 输出精简的原始坐标JSON（无YOLO格式）
3. 生成VideoLLAMA3可直接使用的conversation输入文件
4. 支持关键帧采样和轨迹断裂处理
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
from tqdm import tqdm


class VideoLLAMA3Preprocessor:
    def __init__(self, image_folder: str, annotation_path: str, output_dir: str, target_id: int):
        """
        初始化预处理器
        
        Args:
            image_folder: 原始图像序列文件夹路径
            annotation_path: 原始标注文件路径
            output_dir: 输出目录
            target_id: 目标ID
        """
        self.image_folder = Path(image_folder)
        self.annotation_path = annotation_path
        self.output_dir = Path(output_dir) / str(target_id)
        self.target_id = target_id
        
        # 创建输出子目录
        self.video_dir = self.output_dir / "video"
        self.json_dir = self.output_dir / "json"
        self.image_dir = self.output_dir / "images"
        
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取图像文件列表
        self.image_files = self._get_image_files()
        
        # 加载标注
        self.annotations = self._load_annotations()
        
        # 帧号偏移（标注从1开始，图像从0开始）
        self.frame_offset = 1
        
    def _get_image_files(self) -> Dict[int, Path]:
        """获取图像序列文件夹中的所有图像文件"""
        image_files = {}
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
        
        for ext in extensions:
            for img_path in self.image_folder.glob(ext):
                stem = img_path.stem
                frame_id = self._parse_frame_id(stem)
                if frame_id is not None:
                    image_files[frame_id] = img_path
        
        if not image_files:
            raise ValueError(f"在 {self.image_folder} 中未找到任何图像文件")
        
        print(f"找到 {len(image_files)} 张图像，帧范围: {min(image_files.keys())} ~ {max(image_files.keys())}")
        return image_files
    
    def _parse_frame_id(self, filename: str) -> Optional[int]:
        """从文件名中解析帧号"""
        import re
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])
        return None
    
    def _load_annotations(self) -> Dict[int, List[Dict]]:
        """加载标注文件"""
        annotations_by_frame = defaultdict(list)
        
        if self.annotation_path.endswith('.txt'):
            with open(self.annotation_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 6:
                        frame_id = int(parts[0])
                        target_id = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        w = float(parts[4])
                        h = float(parts[5])
                        
                        annotations_by_frame[frame_id].append({
                            'id': target_id,
                            'bbox': [x, y, w, h]
                        })
        
        print(f"加载标注完成，共 {len(annotations_by_frame)} 帧包含标注")
        return dict(annotations_by_frame)
    
    def get_target_timeline(self, start_frame: int, end_frame: int) -> List[Tuple[int, int]]:
        """获取目标在指定时间范围内出现的连续片段"""
        segments = []
        current_segment = None
        
        for frame in range(start_frame, end_frame + 1):
            has_target = False
            if frame in self.annotations:
                for ann in self.annotations[frame]:
                    if ann['id'] == self.target_id:
                        has_target = True
                        break
            
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
    
    def extract_bbox_annotations(self, start_frame: int, end_frame: int, 
                                   sample_interval: int = 1) -> Dict[int, List[int]]:
        """
        提取目标在指定范围内的bbox标注（原始坐标）
        
        Args:
            start_frame: 起始帧（标注帧号）
            end_frame: 结束帧（标注帧号）
            sample_interval: 采样间隔，1表示每帧都提取
            
        Returns:
            {frame_id: [x, y, w, h]}
        """
        bbox_dict = {}
        
        for frame in range(start_frame, end_frame + 1):
            if frame % sample_interval != 0:
                continue
                
            if frame in self.annotations:
                for ann in self.annotations[frame]:
                    if ann['id'] == self.target_id:
                        bbox_dict[frame] = [int(v) for v in ann['bbox']]
                        break
        
        return bbox_dict
    
    def create_side_by_side_frame(self, frame_id: int, bbox: Optional[List[int]], 
                                   original_frame: np.ndarray, key_frames: Optional[List[int]]=None) -> np.ndarray:
        """
        创建左右拼接帧（原图+放大效果图）
        
        Args:
            frame_id: 帧号
            bbox: [x, y, w, h] 或 None（目标不存在）
            original_frame: 原始图像
            
        Returns:
            拼接后的图像
        """
        h, w = original_frame.shape[:2]
        
        # 左侧：原图 + 细bbox
        left_frame = original_frame.copy()
        
        if bbox is not None:
            x, y, bw, bh = bbox
            # 绘制细bbox（线宽1，红色）
            cv2.rectangle(left_frame, (x, y), (x+bw, y+bh), (0, 0, 255), 1)
            # 在bbox上方标注ID
            cv2.putText(left_frame, f"ID:{self.target_id}", (x, y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 添加帧号
        cv2.putText(left_frame, f"Frame: {frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 右侧：放大效果图
        right_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        if bbox is not None and bw > 0 and bh > 0:
            x, y, bw, bh = bbox
            
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
                    cv2.putText(right_frame, f"Zoom: target ID {self.target_id}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            # 目标不存在时显示提示
            cv2.putText(right_frame, "Target not present", (w//4, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
        
        # 添加分隔线
        separator = np.ones((h, 5, 3), dtype=np.uint8) * 255
        side_by_side = np.hstack([left_frame, separator, right_frame])

        # 把关键帧的bbox可视化原图和拼接了的放大效果图单独保存
        if frame_id in key_frames:
            cv2.imwrite(str(self.image_dir / f"frame_{(frame_id-1):06d}.jpg"), left_frame)
            cv2.imwrite(str(self.image_dir / f"frame_wz_{(frame_id-1):06d}.jpg"),side_by_side)
        
        return side_by_side
    
    def generate_visualization_video(self, start_frame: int, end_frame: int, key_frames_nums: int,
                                      key_frames: Optional[list[int]] = None,
                                      output_name: str = "visualization.mp4",
                                      fps: int = 30, sample_interval: int = 1) -> str:
        """
        生成左右拼接的可视化视频
        
        Args:
            start_frame: 起始帧（标注帧号）
            end_frame: 结束帧（标注帧号）
            key_frames_nums: 用于等间隔采样生成的关键帧数目（用于生成VideoLLAMA3输入的关键帧列表）
            key_frames: 关键帧（标注帧号）
            output_name: 输出视频文件名
            fps: 视频帧率
            sample_interval: 采样间隔
            
        Returns:
            输出视频路径
        """
        # 获取目标出现片段
        segments = self.get_target_timeline(start_frame, end_frame)
        frames_with_target = set()
        for seg_start, seg_end in segments:
            for f in range(seg_start, seg_end + 1):
                frames_with_target.add(f)
        
        # 获取第一帧确定尺寸
        first_valid_frame = start_frame
        while first_valid_frame not in self.image_files:
            first_valid_frame += 1
        
        first_img = cv2.imread(str(self.image_files[first_valid_frame]))
        h, w = first_img.shape[:2]
        
        # 输出视频尺寸（拼接后宽度 = 原图宽 + 5px分隔线 + 原图宽）
        out_w = w * 2 + 5
        out_h = h
        
        output_path = self.video_dir / output_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
        
        print(f"生成可视化视频: {output_path}")
        print(f"目标ID: {self.target_id}, 帧范围: {start_frame} ~ {end_frame}")
        print(f"采样间隔: {sample_interval}, 实际输出帧数: ~{(end_frame-start_frame+1)//sample_interval}")
        print(f"出现片段: {segments}")

        # 生成关键帧列表（等间隔采样 + 用户指定）
        sampled_key_frames = list(np.linspace(start_frame, end_frame, key_frames_nums, dtype=int))
        if key_frames is not None:
            key_frames = list(set(key_frames) | set(map(int,sampled_key_frames)))
            key_frames.sort()
        else:
            key_frames = sampled_key_frames

        for frame_id in tqdm(range(start_frame, end_frame + 1, sample_interval), desc="生成视频"):
            # 获取图像（注意帧号偏移）
            img_frame_id = frame_id - self.frame_offset
            if img_frame_id in self.image_files:
                frame = cv2.imread(str(self.image_files[img_frame_id]))
            else:
                # 创建黑帧
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(frame, f"Frame {frame_id} (MISSING)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # 获取当前帧的bbox
            bbox = None
            if frame_id in self.annotations:
                for ann in self.annotations[frame_id]:
                    if ann['id'] == self.target_id:
                        bbox = [int(v) for v in ann['bbox']]
                        break
            
            # 创建拼接帧
            side_by_side = self.create_side_by_side_frame(frame_id, bbox, frame, key_frames)
            out_writer.write(side_by_side)
        
        out_writer.release()
        print(f"可视化视频已保存: {output_path}")
        return str(output_path)
    
    def export_bbox_json(self, start_frame: int, end_frame: int, 
                         sample_interval: int = 1) -> Dict[str, Any]:
        """
        导出bbox标注为JSON文件
        
        Args:
            start_frame: 起始帧
            end_frame: 结束帧
            sample_interval: 采样间隔
            
        Returns:
            bbox字典
        """
        bbox_dict = self.extract_bbox_annotations(start_frame, end_frame, sample_interval)
        
        output_data = {
            "target_id": self.target_id,
            "frame_range": [start_frame, end_frame],
            "sample_interval": sample_interval,
            "frame_offset_note": "Annotation frames start from 1, image frames start from 0",
            "bbox_format": "[x1, y1, w, h]",
            "annotations": bbox_dict
        }
        
        output_path = self.json_dir / f"target_{self.target_id}_bbox.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"bbox标注已保存: {output_path}")
        return output_path
    
    def generate_videollama3_input(self, video_path: str, bbox_json_path: str,
                                     start_frame: int, end_frame: int, key_frames_nums:int,
                                     key_frames: Optional[List[int]] = None,
                                     max_frames: int = 180) -> Dict:
        """
        生成VideoLLAMA3的conversation输入
        
        Args:
            video_path: 可视化视频路径
            bbox_json_path: bbox标注JSON路径
            start_frame: 起始帧
            end_frame: 结束帧
            key_frames_nums: 用于等间隔采样生成的关键帧数目（用于生成VideoLLAMA3输入的关键帧列表）
            key_frames: 关键帧列表（如行为变化点）
            max_frames: 最大帧数限制
            
        Returns:
            VideoLLAMA3 conversation格式的字典
        """
        # 加载bbox数据
        with open(bbox_json_path, 'r') as f:
            bbox_data = json.load(f)
        
        bbox_dict = bbox_data["annotations"]
        
        # 生成关键帧列表（等间隔采样 + 用户指定）
        sampled_key_frames = list(np.linspace(start_frame, end_frame, key_frames_nums, dtype=int))
        if key_frames is not None:
            key_frames = list(set(key_frames) | set(sampled_key_frames))
            key_frames.sort()
        else:
            key_frames = sampled_key_frames

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
            Target ID: {self.target_id}
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
        output_path = self.json_dir / f"target_{self.target_id}_videollama3_input.json"
        with open(output_path, 'w') as f:
            json.dump(conversation, f, indent=2)
        
        print(f"VideoLLAMA3输入已保存: {output_path}")
        return conversation


def main():
    parser = argparse.ArgumentParser(description='MOT数据预处理（VideoLLAMA3专用版）')
    parser.add_argument('--image_folder', type=str,
                        default="G:/UAVBenchmark/OUR_DATASET/DynUAVI/split/test/009/img1",
                        help='原始图像序列文件夹路径')
    parser.add_argument('--annotation', type=str,
                        default="G:/UAVBenchmark/OUR_DATASET/DynUAVI/split/test/009/gt/gt.txt",
                        help='原始标注文件路径')
    parser.add_argument('--output', type=str,
                        default="G:/language-conditional_MOT/DynUAV/009",
                        help='输出目录')
    parser.add_argument('--target_id', type=int, default=5,
                        help='要提取的目标ID')
    parser.add_argument('--start_frame', type=int, default=1,
                        help='起始帧号（标注帧号）')
    parser.add_argument('--end_frame', type=int, default=207,
                        help='结束帧号（标注帧号）')
    parser.add_argument('--fps', type=int, default=30,
                        help='输出视频帧率')
    parser.add_argument('--video_sample_interval', type=int, default=1,
                        help='制作目标轨迹的视频采样间隔（1表示每帧都采样）')
    parser.add_argument('--key_frames', type=str, default='113,145,161,203',
                        help='关键帧列表，逗号分隔，如: 300,500,700')
    parser.add_argument('--key_frames_nums', type=int, default=20,
                        help='用于等间隔采样生成的关键帧数目（用于生成VideoLLAMA3输入的关键帧列表）')
    parser.add_argument('--max_frames', type=int, default=180,
                        help='VideoLLAMA3最大帧数限制')
    parser.add_argument('--no_video', action='store_true',
                        help='不生成可视化视频')
    args = parser.parse_args()
    
    # 解析关键帧
    key_frames = None
    if args.key_frames:
        key_frames = [int(x.strip()) for x in args.key_frames.split(',')]
    
    # 初始化预处理器
    preprocessor = VideoLLAMA3Preprocessor(
        args.image_folder, args.annotation, args.output, args.target_id
    )
    
    # 1. 导出bbox JSON
    bbox_json_path = preprocessor.export_bbox_json(args.start_frame,\
                         args.end_frame, args.video_sample_interval)
    
    # 2. 生成可视化视频
    video_path = None
    if not args.no_video:
        video_path = preprocessor.generate_visualization_video(
            args.start_frame, args.end_frame, args.key_frames_nums, key_frames=key_frames,
            output_name=f"target_{args.target_id}_vis.mp4", fps=args.fps,
            sample_interval=args.video_sample_interval
        )
    
    # 3. 生成VideoLLAMA3输入
    if video_path:
        preprocessor.generate_videollama3_input(
            video_path, str(bbox_json_path),
            args.start_frame, args.end_frame,
            args.key_frames_nums,
            key_frames=key_frames,
            max_frames=args.max_frames
        )
    
    print(f"\n=== 处理完成 ===")
    print(f"输出目录: {preprocessor.output_dir}")
    print(f"  - 视频: {preprocessor.video_dir}")
    print(f"  - JSON: {preprocessor.json_dir}")


if __name__ == "__main__":
    main()