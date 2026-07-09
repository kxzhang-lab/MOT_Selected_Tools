from openai import OpenAI
import argparse
import os
from pathlib import Path
import json
import ast

os.environ['DASHSCOPE_API_KEY'] = ''  # 清除环境变量干扰
os.environ['MODELSCOPE_API_TOKEN'] = ''

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
    
    def process_trajectory(self, trajectory_dir, sample_num, output_dir):
        """处理目标轨迹图像并生成描述
        Args:
            trajectory_dir (str): 目标轨迹图像所在目录
            output_dir (Path): 结果保存文件夹
        Returns:
            str: 模型生成的描述文本
        """
        # 构建对话输入
        conversation = build_conversation(trajectory_dir, sample_num)
        # 运行推理
        response = self.run_inference([conversation])
        # 保存响应
        parse_result = parse_with_json5(response,Path(trajectory_dir).parts[-1])  # 解析响应，提取有用信息
        if parse_result["json_content"]:
            output_path = output_dir / "target_description.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parse_result, f, ensure_ascii=False, indent=4)
            print(f"轨迹——{os.path.basename(trajectory_dir)}——的结果已保存到: {output_path}")
        
def get_trajectory_directory(dataset_entry, video_sequence, target_id):
    """构建目标轨迹图像目录路径
    Args:
        dataset_entry (str): 数据集入口路径
        video_sequence (str): 视频序列ID
        target_id (str): 目标ID
    Returns:
        str: 目标轨迹图像目录路径
    """
    trajectory_dir = f"{dataset_entry}/{video_sequence}/images/{target_id}"
    if not os.path.isdir(trajectory_dir):
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")
    return trajectory_dir

def make_output_directionary(dataset_entry:str, video_sequence:str, target_id:str):
    """构建描述存储路径

    Args:
        dataset_entry (str): 数据集入口路径
        video_sequence (str): 视频序号
        target_id (str): 轨迹ID号
    return:
        output_dir (Path): 构建的输出路径
    """
    output_dir = Path(f"{dataset_entry}/{video_sequence}/json/Qwen3.5-35B-A3B/{target_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def single_trajectory_process(inference:QwenVLMInference, dataset_entry:str, 
                              video_sequence:str, target_id:str, sample_num:int):
    """单条轨迹处理函数
    Params:
        inference (QwenVLMInference): Qwen大模型的前向推理类
        dataset_entry (str): 数据集入口路径
        video_sequence (str): 视频序号
        target_id (str): 轨迹ID号
    """
    trajectory_dir = get_trajectory_directory(dataset_entry, video_sequence, target_id)  # 获取目标轨迹图像目录路径  
    output_dir = make_output_directionary(dataset_entry, video_sequence, target_id)
    inference.process_trajectory(trajectory_dir=trajectory_dir, sample_num=sample_num, output_dir=output_dir)  # 执行轨迹前向推理

def batch_trajectory_process(inference:QwenVLMInference, dataset_entry:str, video_sequence:str, sample_num:int):
    """轨迹批处理流程
    Params:
        inference (QwenVLMInference): Qwen大模型的前向推理类
        dataset_entry (str): 数据集入口路径
        video_sequence (str): 视频序号
    """
    all_trajectories = [target_id for target_id in os.listdir(f"{dataset_entry}/{video_sequence}/images") 
                        if (target_id.isdigit() and os.path.isdir(f"{dataset_entry}/{video_sequence}/images/{target_id}"))]
    all_trajectories.sort(key=lambda x:ast.literal_eval(x))
    for target_id in all_trajectories: 
        single_trajectory_process(inference, dataset_entry, video_sequence, target_id, sample_num)
    
def main():
    args = make_parse()  # 读取输入参数 
    inference = QwenVLMInference(api_key=args.api_key, base_url=args.base_url, 
                                 model_id=args.model_id)  # 初始化QwenVLMInference实例
    if args.target_id and len(args.target_id) > 0:  # 指定了轨迹ID那就只描述这一个轨迹
        single_trajectory_process(inference, args.dataset_entry, args.video_sequence, args.target_id, args.sample_num)
    else:  # 要是没有指定轨迹ID那就描述该视频下的所有轨迹
        batch_trajectory_process(inference, args.dataset_entry, args.video_sequence, args.sample_num)

def make_parse():
    parser = argparse.ArgumentParser(description="Run Qwen3.5-35B-A3B on target trajectory")
    parser.add_argument("--api_key", type=str, default='ms-47969833-d311-40fe-90a9-138df34652ac', help="API key for authentication")
    parser.add_argument("--base_url", type=str, default='https://api-inference.modelscope.cn/v1', help="Base URL for the API")
    parser.add_argument("--model_id", type=str, default='Qwen/Qwen3.5-35B-A3B', help="Model ID to use for inference")
    parser.add_argument("--sample_num", type=int, default=19)  # 构建data_url时的采样间隔
    parser.add_argument("--dataset_entry", type=str, default='./data/DynUAVI')  # 数据集入口路径
    parser.add_argument("--video_sequence", type=str, default='061')  # 视频序列ID
    parser.add_argument("--target_id", type=str, default='')  # 目标ID
    return parser.parse_args()

if __name__=='__main__':
    main()
