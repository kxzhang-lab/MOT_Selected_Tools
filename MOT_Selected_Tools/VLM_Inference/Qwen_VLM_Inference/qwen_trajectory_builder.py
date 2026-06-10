import glob
import re
from scripts.Qwen_VLM_Inference.qwen_image_processor import build_image_content

def build_trajectory_prompt():
    """构建目标轨迹的描述语料
    """
    image_description = """
    You are generating structured annotations for language-conditioned uav multi-object tracking.
    The red bounding box indicates the target.\n
    These images are sampled from the same target trajectory in chronological order.\n
    """
    description_demands = """
    1. Fill the following fixed schema.\n
    2. Do not add new fields.\n
    3. Use \"uncertain\" if not visible.\n
    4. Avoid subjective guesses.\n
    5. Prefer scene-grounded landmarks over image coordinates.\n
    """
    output_demands = """
    Output only valid JSON:\n
    {
        'target_type': \"\"
        'appearance': {
            'main_color': \"\",
            'secondary_color': \"\",
            'object_subtype': \"\",
        },
        'movement_path':{
            'start_region':\"\",
            'passed_landmarks':[],
            'end_region':\"\",
        },
        'behavior_sequence':[
            {
                'stage':\"early\",
                'behavior':\"\",
                'speed_pattern':\"\",
                'spatial_region':\"\",
            },
            {
                'stage':\"middle\",
                'behavior':\"\",
                'speed_pattern':\"\",
                'spatial_region':\"\",
            },
            {
                'stage':\"late\",
                'behavior':\"\",
                'speed_pattern':\"\",
                'spatial_region':\"\",
            },
        ],
        'spatial_landmarks': [],
        'uncertain_items': []
    }
    """
    target_type_list = """
    Fixed target type list:\n
    car, people, cycler, truck, excavator, crane, bus, cycle"""
    object_subtype_list = """
    Fixed object subtype list:
    For people: t-shirt, long-sleeve_top, pants, dress, robe, uncertain.
    For cycler: bicycle_rider, electric_bike_rider, motorcycle_rider, scooter_rider, uncertain.
    For cycle: bicycle, electric_bike, motorcycle, scooter, uncertain.
    For car: sedan, suv, van, taxi, tricycle, uncertain.
    For truck: cargo_truck, box_truck, dump_truck, uncertain.
    For bus: bus, uncertain.
    For excavator: excavator, uncertain.
    For crane: crane, uncertain.
    Choose one subtype according to the target_type. If uncertain, use "uncertain".
    """
    color_list = """
    Fixed color list:\n
    black, white, red,blue,yellow,green,grey,brown,uncertain.\n
    """
    behavior_list = """
    Fixed behavior list:\n
    standing, sitting, walking, running, cycling, driving, stopping, turning, climbing, entering, leaving, operating, uncertain.\n
    """
    speed_pattern_list = """
    Fixed speed pattern list:\n
    stationary, steady, speed_up, slow_down, stop_then_move, move_then_stop, uncertain
    """
    good_example = """
    Good example:\n
    road beside the building\n
    grey stair-like structure\n
    area near the excavator\n
    riverside walkway\n
    parking area beside the trees\n
    """
    bad_example="""
    Bad example:\n
    left side of image\n
    upper-right corner\n
    middle of frame\n
    bottom area\n
    """
    spational_description_rules = f"""
    Spatial Description Rules:\n
    1. Use scene-grounded landmarks whenever possible.\n
    2. Avoid descriptions based solely on image coordinates.\n
       (left/right/top/bottom).\n
    3. Only include scene objects or structures that help identify the target.\n
    4. Avoid generic background descriptions unless they distinguish the target from nearby objects.\n
    5. Prefer semantic locations over pixel locations.\n
    6. Use the same landmark consistently across stages if it refers to the same structure.\n
    \n
    {good_example}\n
    {bad_example}\n
    """
    output_formulation_rules = """
    Output Formulation Rules:\n
    1. Output only one JSON object.\n
    2. The JSON object must be enclosed by:
    ```json
    {
        ...
    }
    ```
    3. Do not output explanations before or after the JSON block.\n
    4. Do not output markdown text except the JSON block.\n
    5. If any field is uncertain, fill the value with 'uncertain'. Do not omit fields.\n
    """
    prompt = f"""
    {image_description}\n
    {description_demands}\n
    {target_type_list}\n
    {object_subtype_list}\n
    {color_list}\n
    {behavior_list}\n
    {speed_pattern_list}\n
    {output_demands}\n
    {spational_description_rules}\n
    {output_formulation_rules}\n
    """
    return prompt

def build_conversation(trajectory_dir, sample_num):
    """构建包含图像输入的对话列表
    Args:
        trajectory_dir (str): 目标轨迹图像所在目录
        sample_num (int): 被用来制作image_url的image数目
    Returns:
        list: 包含文本和图像输入的对话列表
    """
    all_image_files = sorted(glob.glob(f"{trajectory_dir}/images/frame_*.jpg"))
    # 只保留 frame_ 后面紧跟6位数字的文件
    pattern = re.compile(r'frame_\d{6}\.jpg$')
    image_files = [f for f in all_image_files if pattern.search(f)]
    if not image_files:
        raise FileNotFoundError(f"No images found in {trajectory_dir}/images/")
    image_content = build_image_content(image_files, sample_num)  # 采样所有图像
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