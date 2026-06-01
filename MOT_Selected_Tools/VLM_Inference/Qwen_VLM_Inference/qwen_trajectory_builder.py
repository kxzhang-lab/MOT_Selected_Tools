import glob
import re
from scripts.Qwen_VLM_Inference.qwen_image_processor import build_image_content

def build_trajectory_prompt():
    """构建目标轨迹的描述语料
    """
    image_description = """
    The red bounding box indicates the target.\n
    These images are sampled from the same target trajectory in chronological order.\n
    """
    description_demands = """
    Please describe:\n
    1. Appearance of the target.\n
    2. Movement path.\n
    3. Behavior changes over time.\n
    4. Spatial landmarks used to identify the target.\n
    """
    output_demands = """
    Output:\n
    Chinese description\n
    English description\n
    JSON\n
    """
    prompt = f"""
    {image_description}
    {description_demands}
    {output_demands}
    """
    return prompt

def build_conversation(trajectory_dir):
    """构建包含图像输入的对话列表
    Args:
        trajectory_dir (str): 目标轨迹图像所在目录
    Returns:
        list: 包含文本和图像输入的对话列表
    """
    all_image_files = sorted(glob.glob(f"{trajectory_dir}/images/frame_*.jpg"))
    # 只保留 frame_ 后面紧跟6位数字的文件
    pattern = re.compile(r'frame_\d{6}\.jpg$')
    image_files = [f for f in all_image_files if pattern.search(f)]
    if not image_files:
        raise FileNotFoundError(f"No images found in {trajectory_dir}/images/")
    image_content = build_image_content(image_files, sample_interval=1)  # 采样所有图像
    prompt = build_trajectory_prompt()
    conversation = {
        "role": "user",
        "content": image_content + [
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
    return conversation