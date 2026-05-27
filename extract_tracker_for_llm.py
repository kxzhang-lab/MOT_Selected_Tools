"""
航拍MOT数据集预处理脚本（图像序列版）
功能：
1. 根据目标ID和时间戳范围，提取对应的图像帧和标注
2. 生成带bbox可视化的视频，便于快速预览
3. 输出结构化的标注文件，供大模型生成语言描述
"""

import os
import cv2
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from tqdm import tqdm


class MOTExtractor:
    def __init__(self, image_folder: str, annotation_path: str, output_dir: str, target_id: int):
        """
        初始化提取器（图像序列版）
        
        Args:
            image_folder: 原始图像序列文件夹路径（内含 frame_000001.jpg 等文件）
            annotation_path: 原始标注文件路径（支持MOT Challenge格式的txt或json）
            output_dir: 输出目录
        """
        self.image_folder = Path(image_folder)
        self.annotation_path = annotation_path
        self.output_dir = Path(output_dir)
        
        # 创建输出子目录
        self.images_dir = self.output_dir / str(target_id) / "images"
        self.labels_dir = self.output_dir / str(target_id) / "labels"
        self.video_dir = self.output_dir / str(target_id) / "video"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取图像文件列表
        self.image_files = self._get_image_files()
        
        # 加载标注
        self.annotations = self._load_annotations()
        
    def _get_image_files(self) -> Dict[int, Path]:
        """
        获取图像序列文件夹中的所有图像文件
        
        Returns:
            {frame_id: image_path}
        """
        image_files = {}
        
        # 支持的图像格式
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
        
        for ext in extensions:
            for img_path in self.image_folder.glob(ext):
                # 尝试从文件名中解析帧号
                stem = img_path.stem
                frame_id = self._parse_frame_id(stem)
                if frame_id is not None:
                    image_files[frame_id] = img_path
        
        if not image_files:
            raise ValueError(f"在 {self.image_folder} 中未找到任何图像文件")
        
        print(f"找到 {len(image_files)} 张图像，帧范围: {min(image_files.keys())} ~ {max(image_files.keys())}")
        return image_files
    
    def _parse_frame_id(self, filename: str) -> Optional[int]:
        """
        从文件名中解析帧号
        支持多种命名格式：
        - frame_000001.jpg
        - 000001.jpg
        - img_1.jpg
        - 1.jpg
        """
        import re
        
        # 尝试提取纯数字
        numbers = re.findall(r'\d+', filename)
        if numbers:
            # 取最后一个数字串（通常帧号在末尾）
            return int(numbers[-1])
        return None
    
    def _load_annotations(self) -> Dict[int, List[Dict]]:
        """
        加载标注文件，支持多种格式
        
        Returns:
            {frame_id: [{'id': target_id, 'bbox': [x,y,w,h], ...}]}
        """
        annotations_by_frame = defaultdict(list)
        
        # 尝试MOT Challenge txt格式（标准格式）
        # 格式：frame, id, x, y, w, h, conf, class, vis
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
                        conf = float(parts[6]) if len(parts) > 6 else 1.0
                        
                        annotations_by_frame[frame_id].append({
                            'id': target_id,
                            'bbox': [x, y, w, h],
                            'conf': conf
                        })
        
        # 尝试json格式
        elif self.annotation_path.endswith('.json'):
            with open(self.annotation_path, 'r') as f:
                data = json.load(f)
                if 'annotations' in data:
                    for ann in data['annotations']:
                        frame_id = ann.get('frame_id') or ann.get('image_id')
                        if frame_id is not None:
                            annotations_by_frame[int(frame_id)].append(ann)
                else:
                    for frame_id, anns in data.items():
                        annotations_by_frame[int(frame_id)] = anns
        
        print(f"加载标注完成，共 {len(annotations_by_frame)} 帧包含标注")
        return dict(annotations_by_frame)
    
    def get_target_timeline(self, target_id: int, start_frame: int, end_frame: int) -> List[Tuple[int, int]]:
        """
        获取目标在指定时间范围内出现的连续片段
        
        Args:
            target_id: 目标ID
            start_frame: 起始帧
            end_frame: 结束帧
            
        Returns:
            [(seg_start, seg_end), ...] 连续出现的片段列表
        """
        segments = []
        current_segment = None
        
        for frame in range(start_frame, end_frame + 1):
            has_target = False
            if frame in self.annotations:
                for ann in self.annotations[frame]:
                    if ann['id'] == target_id:
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
    
    def extract_frames_and_annotations(self, target_id: int, start_frame: int, end_frame: int,
                                        output_prefix: str = "extract") -> Dict:
        """
        提取指定范围内的帧和标注（直接复制图像文件）
        
        Args:
            target_id: 目标ID
            start_frame: 起始帧
            end_frame: 结束帧
            output_prefix: 输出文件前缀
            
        Returns:
            提取信息字典
        """
        # 获取目标出现的连续片段
        segments = self.get_target_timeline(target_id, start_frame, end_frame)
        
        extract_info = {
            'target_id': target_id,
            'original_image_folder': str(self.image_folder),
            'extracted_frames': [],
            'segments': segments,
            'annotations': {}
        }
        
        # 获取可用帧范围
        available_frames = set(self.image_files.keys())
        
        for frame_id in tqdm(range(start_frame, end_frame + 1), desc="提取图像帧"):
            # 检查图像是否存在
            if (frame_id-1) not in available_frames:
                print(f"警告：帧 {frame_id} 图像不存在，跳过")
                continue
            
            # 获取当前帧的标注（仅包含目标ID）
            frame_annotations = []
            if frame_id in self.annotations:
                for ann in self.annotations[frame_id]:
                    if ann['id'] == target_id:
                        frame_annotations.append(ann)
                        break
            
            # 复制图像文件
            src_img_path = self.image_files[frame_id-1]
            img_filename = f"{output_prefix}_frame_{(frame_id-1):06d}{src_img_path.suffix}"
            dst_img_path = self.images_dir / img_filename
            shutil.copy2(src_img_path, dst_img_path)
            
            # 保存标注（YOLO格式）
            if frame_annotations:
                label_filename = f"{output_prefix}_frame_{(frame_id-1):06d}.txt"
                label_path = self.labels_dir / label_filename
                
                # 读取图像尺寸
                img = cv2.imread(str(src_img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    with open(label_path, 'w') as f:
                        for ann in frame_annotations:
                            x, y, bw, bh = ann['bbox']
                            cx = (x + bw/2) / w
                            cy = (y + bh/2) / h
                            nw = bw / w
                            nh = bh / h
                            f.write(f"{target_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            
            extract_info['extracted_frames'].append(frame_id)
            if frame_annotations:
                extract_info['annotations'][frame_id] = frame_annotations
        
        # 保存提取信息JSON
        info_path = self.output_dir / f"{output_prefix}_info.json"
        with open(info_path, 'w') as f:
            json.dump(extract_info, f, indent=2, default=str)
        
        print(f"\n提取完成！")
        print(f"- 提取帧数: {len(extract_info['extracted_frames'])}")
        print(f"- 目标出现帧数: {len(extract_info['annotations'])}")
        print(f"- 出现片段: {segments}")
        
        return extract_info
    
    def visualize_video(self, target_id: int, start_frame: int, end_frame: int,
                        output_name: str = "visualization.mp4", fps: int = 30,
                        show_other_bboxes: bool = True):
        """
        生成带bbox可视化的视频
        
        Args:
            target_id: 目标ID
            start_frame: 起始帧
            end_frame: 结束帧
            output_name: 输出视频文件名
            fps: 输出视频帧率
            show_other_bboxes: 是否显示其他目标的bbox
        """
        # 获取目标出现的连续片段
        segments = self.get_target_timeline(target_id, start_frame, end_frame)
        frames_with_bbox = set()
        for seg_start, seg_end in segments:
            for f in range(seg_start, seg_end + 1):
                frames_with_bbox.add(f)
        
        # 准备输出视频
        output_path = self.video_dir / output_name
        
        # 获取第一帧来确定视频尺寸
        first_frame_id = start_frame-1
        while first_frame_id not in self.image_files and first_frame_id <= end_frame:
            first_frame_id += 1
        if first_frame_id > end_frame:
            raise ValueError("指定范围内没有可用的图像帧")
        
        first_img = cv2.imread(str(self.image_files[first_frame_id]))
        if first_img is None:
            raise ValueError(f"无法读取第一帧图像: {self.image_files[first_frame_id]}")
        
        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        print(f"生成可视化视频: {output_path}")
        print(f"目标ID: {target_id}, 帧范围: {start_frame} ~ {end_frame}")
        print(f"出现片段: {segments}")
        
        for frame_id in tqdm(range(start_frame, end_frame + 1), desc="生成视频"):
            # 读取图像
            if (frame_id-1) not in self.image_files:
                # 如果图像不存在，创建黑帧
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(frame, f"Frame {frame_id-1} (MISSING)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                frame = cv2.imread(str(self.image_files[frame_id-1]))
                if frame is None:
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 获取当前帧中所有标注
            all_bboxes = []
            if frame_id in self.annotations:
                for ann in self.annotations[frame_id]:
                    all_bboxes.append(ann)
            
            # 绘制bbox
            for ann in all_bboxes:
                if not show_other_bboxes and ann['id'] != target_id:
                    continue
                    
                x, y, bw, bh = [int(v) for v in ann['bbox']]
                obj_id = ann['id']
                
                # 目标ID用红色高亮，其他用绿色
                if obj_id == target_id:
                    color = (0, 0, 255)  # 红色
                    thickness = 3
                    cv2.putText(frame, f"ID:{obj_id}", (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    color = (0, 255, 0)  # 绿色
                    thickness = 1
                    cv2.putText(frame, f"ID:{obj_id}", (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, thickness)
            
            # 添加帧号信息
            cv2.putText(frame, f"Frame: {frame_id-1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加出现片段提示
            if frame_id in frames_with_bbox:
                cv2.putText(frame, "TARGET PRESENT", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            out_writer.write(frame)
        
        out_writer.release()
        print(f"可视化视频已保存: {output_path}")
        return str(output_path)
    

def main():
    parser = argparse.ArgumentParser(description='MOT数据提取与可视化工具（图像序列版）')
    parser.add_argument('--image_folder', type=str, default="G:/UAVBenchmark/OUR_DATASET/DynUAVI/split/val/001/img1",
    help='原始图像序列文件夹路径（内含 frame_000001.jpg 等文件）')
    parser.add_argument('--annotation', type=str, default="G:/UAVBenchmark/OUR_DATASET/DynUAVI/split/val/001/gt/gt.txt",
    help='原始标注文件路径（支持MOT Challenge格式的txt或json）')
    parser.add_argument('--output', type=str, default="G:/language-conditional_MOT/DynUAV/001",
    help='输出目录')
    parser.add_argument('--target_id', type=int, default=22,
    help='要提取的目标ID')
    parser.add_argument('--start_frame', type=int, default=261,
    help='起始帧号')
    parser.add_argument('--end_frame', type=int, default=2154,
    help='结束帧号')
    parser.add_argument('--fps', type=int, default=30,
    help='输出视频帧率')
    parser.add_argument('--no_video', action='store_true',
    help='不生成可视化视频')
    parser.add_argument('--no_other_bboxes', default=True,
    help='可视化时不显示其他目标的bbox')
    parser.add_argument('--generate_prompt', action='store_true',
    help='生成供大模型使用的prompt')

    args = parser.parse_args()

    # 初始化提取器
    extractor = MOTExtractor(args.image_folder, args.annotation, args.output, args.target_id)

    # 提取帧和标注
    print(f"\n开始提取目标ID={args.target_id}，帧范围={args.start_frame}~{args.end_frame}")
    extract_info = extractor.extract_frames_and_annotations(
    args.target_id, args.start_frame, args.end_frame,
    output_prefix=f"target_{args.target_id}"
    )

    print(f"\n输出目录: {extractor.output_dir}")
    print(f"- 图像目录: {extractor.images_dir}")
    print(f"- 标注目录: {extractor.labels_dir}")

    # 生成可视化视频
    if not args.no_video:
        video_path = extractor.visualize_video(
        args.target_id, args.start_frame, args.end_frame,
        output_name=f"target_{args.target_id}_vis.mp4",
        fps=args.fps,
        show_other_bboxes=not args.no_other_bboxes
        )
        print(f"\n可视化视频: {video_path}")
    

if __name__ == "__main__":
    main()