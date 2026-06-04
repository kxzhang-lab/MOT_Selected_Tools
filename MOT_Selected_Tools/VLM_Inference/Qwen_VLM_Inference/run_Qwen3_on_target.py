from openai import OpenAI
import argparse
import os
from pathlib import Path
import json

import sys
workspace_dir = Path(__file__).parents[2]  # 获取当前脚本所在目录的上两级目录
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))  # 将上两级目录添加到系统路径，以便导入同目录下的模块
from scripts.Qwen_VLM_Inference.qwen_trajectory_builder import build_conversation
from scripts.Qwen_VLM_Inference.response_process import parse_with_json5

class QwenVLMInference:
    """千问视觉语言模型推理示例
    """
    def __init__(self, api_key, base_url, model_id):
        """初始化QwenVLMInference实例"""
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id

    def run_inference(self, conversations):
        """运行推理并返回结果
        Args:
            conversations (list): 包含文本和图像输入的对话列表
        Returns:
            str: 模型生成的响应文本
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=conversations,
            stream=False
        )
        return response.choices[0].message.content
    
    def process_trajectory(self, trajectory_dir):
        """处理目标轨迹图像并生成描述
        Args:
            trajectory_dir (str): 目标轨迹图像所在目录
        Returns:
            str: 模型生成的描述文本
        """
        # 构建对话输入
        conversation = build_conversation(trajectory_dir)
        # 运行推理
        response = self.run_inference([conversation])
        # 保存结果
        output_dir = Path(trajectory_dir) / "json" / "Qwen3.5-35B-A3B"
        output_dir.mkdir(parents=True, exist_ok=True)
        # 保存响应
        parse_result = parse_with_json5(response)  # 解析响应，提取有用信息
        output_path = output_dir / "target_description.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parse_result, f, ensure_ascii=False, indent=4)
        print(f"结果已保存到: {output_path}")
        
def get_trajectory_directory(dataset_entry, video_sequence, target_id):
    """构建目标轨迹图像目录路径
    Args:
        dataset_entry (str): 数据集入口路径
        video_sequence (str): 视频序列ID
        target_id (str): 目标ID
    Returns:
        str: 目标轨迹图像目录路径
    """
    trajectory_dir = f"{dataset_entry}/{video_sequence}/{target_id}"
    if not os.path.isdir(trajectory_dir):
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")
    return trajectory_dir

def main():
    args = make_parse()  # 读取输入参数
    inference = QwenVLMInference(api_key=args.api_key, base_url=args.base_url, model_id=args.model_id)  # 初始化QwenVLMInference实例
    trajectory_dir = get_trajectory_directory(args.dataset_entry, args.video_sequence, args.target_id)  # 获取目标轨迹图像目录路径  
    inference.process_trajectory(trajectory_dir=trajectory_dir)  # 执行轨迹前向推理

def make_parse():
    parser = argparse.ArgumentParser(description="Run Qwen3.5-35B-A3B on target trajectory")
    parser.add_argument("--api_key", type=str, default='ms-13172eb6-370d-4b42-996d-c8d2b830b661', help="API key for authentication")
    parser.add_argument("--base_url", type=str, default='https://api-inference.modelscope.cn/v1', help="Base URL for the API")
    parser.add_argument("--model_id", type=str, default='Qwen/Qwen3.5-35B-A3B', help="Model ID to use for inference")
    parser.add_argument("--dataset_entry", type=str, default='./data/DynUAV')  # 数据集入口路径
    parser.add_argument("--video_sequence", type=str, default='009')  # 视频序列ID
    parser.add_argument("--target_id", type=str, default='5')  # 目标ID
    return parser.parse_args()

if __name__=='__main__':
    main()
