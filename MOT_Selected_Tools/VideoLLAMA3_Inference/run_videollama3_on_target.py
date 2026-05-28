"""
run_videollama3_on_target.py
功能：读取预处理生成的 conversation 文件，调用 VideoLLAMA3 模型生成描述，
     并将结果保存回原目录（支持中英文描述和JSON格式）
"""

import os
import re
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm


def convert_windows_to_linux_path(windows_path: str, 
                                   mapping: Optional[Dict[str, str]] = None) -> str:
    """
    简单的 Windows 到 Linux 路径转换函数
    
    Args:
        windows_path: Windows 格式路径（如 G:/folder/file.mp4 或 G:\folder\file.mp4）
        mapping: 可选的自定义映射字典，如 {"G:/language-conditional_MOT": "/home/user/data"}
    
    Returns:
        Linux 格式路径
    """
    if not windows_path:
        return windows_path
    
    # 统一将反斜杠转换为正斜杠
    normalized = windows_path.replace('\\', '/')
    
    # 使用自定义映射（如果有）
    if mapping:
        for win_prefix, linux_prefix in mapping.items():
            win_prefix_norm = win_prefix.replace('\\', '/')
            if normalized.startswith(win_prefix_norm):
                return linux_prefix + normalized[len(win_prefix_norm):]
    
    # 自动转换盘符：G:/path -> /mnt/g/path
    drive_match = re.match(r'^([A-Za-z]):/', normalized)
    if drive_match:
        drive = drive_match.group(1).lower()
        return f"/mnt/{drive}/{normalized[3:]}"
    
    # 无法转换，返回原路径
    print(f"警告: 无法转换路径 {windows_path}，将使用原路径")
    return windows_path


class VideoLLAMA3Inference:
    def __init__(self, model_path: str, device: str = "auto", 
                 use_flash_attention: bool = True, torch_dtype: torch.dtype = torch.bfloat16):
        """
        初始化 VideoLLAMA3 模型和处理器
        
        Args:
            model_path: 模型路径（本地或 HuggingFace 模型名）
            device: 设备（"auto", "cuda", "cpu"）
            use_flash_attention: 是否使用 flash_attention_2
            torch_dtype: 数据类型
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"正在加载模型: {model_path}")
        print(f"设备: {device}, 数据类型: {torch_dtype}")
        
        # 加载模型
        attn_implementation = "flash_attention_2" if use_flash_attention else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print("模型加载完成！")
    
    def run_inference(self, conversation: List[Dict], 
                      max_new_tokens: int = 512,
                      do_sample: bool = False,
                      temperature: float = 0.7) -> str:
        """
        运行推理，生成描述
        
        Args:
            conversation: 对话列表，格式符合 VideoLLAMA3 要求
            max_new_tokens: 最大生成 token 数
            do_sample: 是否采样
            temperature: 采样温度
            
        Returns:
            模型生成的响应文本
        """
        # 处理输入
        inputs = self.processor(
            conversation=conversation,
            return_tensors="pt"
        )
        
        # 移动到 GPU
        inputs = {
            k: v.cuda() if isinstance(v, torch.Tensor) and v.device.type != 'cuda' else v
            for k, v in inputs.items()
        }
        
        # 转换数据类型
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        
        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None
            )
        
        # 解码响应
        response = self.processor.batch_decode(
            output_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        return response
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析模型响应，提取中文描述、英文描述和 JSON 字段
        
        Args:
            response: 模型生成的原始响应文本
            
        Returns:
            解析后的字典，包含 chinese, english, json_fields
        """
        result = {
            "chinese": "",
            "english": "",
            "json_fields": {},
            "raw_response": response
        }
        
        # 尝试从响应中提取各部分
        lines = response.split('\n')
        
        current_section = None
        chinese_lines = []
        english_lines = []
        json_lines = []
        
        in_json = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # 检测章节标题
            if 'Chinese' in line_stripped or '中文' in line_stripped:
                current_section = 'chinese'
                continue
            elif 'English' in line_stripped:
                current_section = 'english'
                continue
            elif 'JSON' in line_stripped or 'json' in line_stripped:
                current_section = 'json'
                in_json = False
                continue
            
            # 收集内容
            if current_section == 'chinese':
                if line_stripped and not line_stripped.startswith('```'):
                    chinese_lines.append(line)
            elif current_section == 'english':
                if line_stripped and not line_stripped.startswith('```'):
                    english_lines.append(line)
            elif current_section == 'json':
                # 检测 JSON 代码块边界
                if line_stripped.startswith('```json'):
                    in_json = True
                    continue
                elif line_stripped.startswith('```'):
                    in_json = False
                    continue
                elif in_json or line_stripped.startswith('{'):
                    json_lines.append(line)
        
        # 组合结果
        if chinese_lines:
            result["chinese"] = '\n'.join(chinese_lines).strip()
        if english_lines:
            result["english"] = '\n'.join(english_lines).strip()
        
        # 尝试解析 JSON
        if json_lines:
            json_str = '\n'.join(json_lines).strip()
            try:
                result["json_fields"] = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    json_str = json_str.replace('```json', '').replace('```', '').strip()
                    result["json_fields"] = json.loads(json_str)
                except json.JSONDecodeError:
                    result["json_fields"] = {"parse_error": json_str[:200]}
        
        return result
    
    def process_target(self, input_json_path: str, output_dir: str,
                       max_new_tokens: int = 512,
                       save_raw_response: bool = True,
                       path_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        处理单个目标的 conversation 文件
        
        Args:
            input_json_path: conversation JSON 文件路径
            output_dir: 输出目录
            max_new_tokens: 最大生成 token 数
            save_raw_response: 是否保存原始响应
            path_mapping: 路径映射字典，如 {"G:/language-conditional_MOT": "/home/user/data"}
            
        Returns:
            生成结果的字典
        """
        # 读取 conversation 文件
        with open(input_json_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
        
        # 提取 conversation 列表
        if "conversation" in conversation_data:
            conversation = conversation_data["conversation"]
        else:
            conversation = conversation_data
        
        # ========== 路径转换（只改这里） ==========
        # 遍历 conversation，找到 video_path 并转换
        for conv in conversation:
            if "content" not in conv:
                continue
            for content_item in conv["content"]:
                if content_item.get("type") == "video":
                    video_dict = content_item.get("video", {})
                    if "video_path" in video_dict:
                        original_path = video_dict["video_path"]
                        new_path = convert_windows_to_linux_path(original_path, path_mapping)
                        video_dict["video_path"] = new_path
                        content_item["video"] = video_dict
                        print(f"路径转换: {original_path} -> {new_path}")
        # ========================================
        
        # 打印视频信息
        if conversation and "content" in conversation[0]:
            for content_item in conversation[0]["content"]:
                if content_item.get("type") == "video":
                    video_info = content_item.get("video", {})
                    print(f"视频路径: {video_info.get('video_path', 'N/A')}")
                    print(f"FPS: {video_info.get('fps', 'N/A')}")
                    print(f"Max frames: {video_info.get('max_frames', 'N/A')}")
                elif content_item.get("type") == "text":
                    text_preview = content_item.get("text", "")[:200]
                    print(f"问题预览: {text_preview}...")

        
        # 运行推理
        print("\n正在生成描述...")
        response = self.run_inference(conversation, max_new_tokens=max_new_tokens)
        
        # 解析响应
        parsed_result = self.parse_response(response)
        
        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 提取目标 ID（从文件名中）
        target_id = None
        if "target_" in str(input_json_path):
            match = re.search(r'target_(\d+)', str(input_json_path))
            if match:
                target_id = int(match.group(1))
        
        # 保存完整结果
        result_data = {
            "target_id": target_id,
            "input_file": str(input_json_path),
            "chinese_description": parsed_result["chinese"],
            "english_description": parsed_result["english"],
            "structured_json": parsed_result["json_fields"],
            "generation_config": {
                "max_new_tokens": max_new_tokens
            }
        }
        
        if save_raw_response:
            result_data["raw_response"] = parsed_result["raw_response"]
        
        # 保存到 JSON 文件
        result_json_path = output_path / f"target_{target_id}_description.json" if target_id else output_path / "description_result.json"
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存: {result_json_path}")
        
        # 打印生成的描述
        print("\n" + "="*60)
        print("生成结果预览")
        print("="*60)
        if parsed_result["chinese"]:
            print(f"\n【中文描述】\n{parsed_result['chinese']}")
        if parsed_result["english"]:
            print(f"\n【英文描述】\n{parsed_result['english']}")
        if parsed_result["json_fields"]:
            print(f"\n【结构化JSON】\n{json.dumps(parsed_result['json_fields'], indent=2, ensure_ascii=False)[:500]}...")
        print("="*60)
        
        return result_data


def batch_process(input_dir: str, output_dir: str, model_path: str,
                  max_new_tokens: int = 512,
                  pattern: str = "*videollama3_input.json",
                  device: str = "auto",
                  path_mapping: Optional[Dict[str, str]] = None) -> List[Dict]:
    """
    批量处理目录下的所有 conversation 文件
    """
    input_path = Path(input_dir)
    json_files = list(input_path.glob(pattern))
    
    if not json_files:
        print(f"未找到匹配 {pattern} 的文件")
        return []
    
    print(f"找到 {len(json_files)} 个文件待处理")
    
    # 初始化模型（只初始化一次）
    inference_engine = VideoLLAMA3Inference(model_path, device=device)
    
    results = []
    for json_file in tqdm(json_files, desc="处理进度"):
        try:
            result = inference_engine.process_target(
                str(json_file), output_dir, 
                max_new_tokens=max_new_tokens,
                path_mapping=path_mapping
            )
            results.append(result)
        except Exception as e:
            print(f"处理 {json_file} 时出错: {e}")
            continue
    
    # 保存汇总结果
    summary_path = Path(output_dir) / "batch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n批量处理完成！汇总结果保存至: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='运行 VideoLLAMA3 推理生成目标描述')
    
    # 必需参数
    parser.add_argument('--input', type=str, default="./data/DynUAV/001/48/json/target_48_videollama3_input.json",
                        help='输入文件或目录路径（conversation JSON 文件或包含该文件的目录）')
    parser.add_argument('--output', type=str, default="./data/DynUAV/001/48/json",
                        help='输出目录')
    parser.add_argument('--model_path', type=str, default="./VideoLLaMA3-7B",
                        help='VideoLLAMA3 模型路径')
    
    # 路径映射参数（新增）
    parser.add_argument('--path_mapping', type=str, default="G:\\language-conditional_MOT:./data",
                        help='路径映射，格式: "G:/path:/home/path" 或 "G:/path=/home/path"')
    
    # 可选参数
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='最大生成 token 数')
    parser.add_argument('--pattern', type=str, default="*videollama3_input.json",
                        help='批量处理时的文件匹配模式')
    parser.add_argument('--device', type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help='设备类型')
    parser.add_argument('--no_flash_attention', action='store_true',
                        help='禁用 flash_attention_2')
    parser.add_argument('--save_raw_response', action='store_true',
                        help='保存原始响应文本')
    
    args = parser.parse_args()
    
    # 解析路径映射（新增）
    path_mapping = None
    if args.path_mapping:
        # 支持 "G:/path:/home/path" 或 "G:/path=/home/path" 格式
        mapping_str = args.path_mapping
        # 找到第一个冒号分隔的位置（盘符后的冒号需要特殊处理）
        # 例如 "G:/folder:/home/folder" -> windows部分 "G:/folder", linux部分 "/home/folder"
        for sep in [':', '=']:
            if sep in mapping_str:
                # 找到第二个冒号的位置（第一个冒号是盘符的一部分）
                first_colon = mapping_str.find(':')
                if first_colon != -1:
                    sep_pos = mapping_str.find(sep, first_colon + 1)
                    if sep_pos != -1:
                        windows_part = mapping_str[:sep_pos]
                        linux_part = mapping_str[sep_pos + 1:]
                        path_mapping = {windows_part: linux_part}
                        break
    
    # 判断输入是文件还是目录
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # 单文件处理
        print(f"单文件模式: {input_path}")
        
        inference_engine = VideoLLAMA3Inference(
            args.model_path,
            device=args.device,
            use_flash_attention=not args.no_flash_attention
        )
        
        result = inference_engine.process_target(
            str(input_path),
            str(output_path),
            max_new_tokens=args.max_new_tokens,
            save_raw_response=args.save_raw_response,
            path_mapping=path_mapping
        )
        
    elif input_path.is_dir():
        # 批量处理
        print(f"批量模式: {input_path}")
        batch_process(
            str(input_path),
            str(output_path),
            args.model_path,
            max_new_tokens=args.max_new_tokens,
            pattern=args.pattern,
            device=args.device,
            path_mapping=path_mapping
        )
    else:
        print(f"错误: 输入路径不存在 - {input_path}")
        return


if __name__ == "__main__":
    main()