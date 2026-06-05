# 路径、颜色、采样参数等配置
from pathlib import Path
import os

FPS = 30  # 输出视频帧率
SAMPLE_INTERVAL = 1  # 制作目标轨迹的视频采样间隔（1表示每帧都采样）
KEY_FRAME_NUMS = 24  # 等间隔采样的关键帧数
FRAME_OFFSET = 1  # 帧号偏移（标注从1开始，图像从0开始）

def get_image_anno_path(dataset_entry, dataset_name, 
                        split_name, video_name):
    """
    根据数据集名称、子集名称和视频名称获取图像序列路径和标注路径
    Params:
        dataset_entry: 数据集总路径
        dataset_name: 数据集名称
        split_name: 子集名称
        video_name: 视频名称
    return:
        image_folder: 图像序列路径所在文件夹名称
        annotation_path: 标注路径
    """
    image_folder, annotation_path = None, None
    if 'DynUAV' in dataset_name:
        image_folder = Path(f"{dataset_entry}") / "OUR_DATASET" / f"{dataset_name}" \
              / "split" / f"{split_name}" / f"{video_name}" / "img1"
        annotation_path = Path(f"{dataset_entry}") / "OUR_DATASET" / f"{dataset_name}" \
              / "split" / f"{split_name}" / f"{video_name}" / "gt" / "gt.txt"
    elif 'VisDrone' in dataset_name:
        visdrone_folder = 'VisDrone2019-MOT'
        ## 注意：VisDrone的test-Challenge集里面的视频数据没有对应标注文件
        annotation_path = Path(f"{dataset_entry}") / f"{visdrone_folder}" / f"{visdrone_folder}" \
              / f"{split_name}" / f"annotations" / f"{video_name}.txt"
        if not annotation_path.exists():
            raise FileExistsError(f"video annotation in {split_name} of {dataset_name} doesn't exist.")
        image_folder = Path(f"{dataset_entry}") / f"{visdrone_folder}" / f"{visdrone_folder}" \
              / f"{split_name}" / f"sequences" / f"{video_name}"
    elif 'UAVDT' in dataset_name:
        image_folder = Path(f"{dataset_entry}") / f"{dataset_name}" / "UAV-benchmark" \
              / f"UAV-benchmark-M" / f"{video_name}" 
        annotation_path = Path(f"{dataset_entry}") / f"{dataset_name}" / "UAV-benchmark" \
              / f"UAV-benchmark-MOTD_v1.0" / f"GT" / f"{video_name}_gt.txt"
    else:
        raise ValueError(f"no valid dataset name.")
    return str(image_folder), str(annotation_path)


def make_output_folders(all_output_dir, dataset_name, video_name, target_id,
                        no_video, no_save_frames, no_export_json):
    """
    创建该视频的相关输出路径
    Params:
        all_output_dir: 总输出路径
        dataset_name: 数据集名称
        video_name: 视频名称
        target_id: 待可视化的目标id
        no_video: 输出bbox可视化拼接视频的标志位
        no_save_frames: 输出bbox在关键帧上可视化的标志位
        no_export_json: 不导出关键帧bbox坐标的标志位
    return:
        output_dict: 输出路径字典。包含：关键帧、视频和坐标.json文件。
    """
    output_dict = {"json":"","images":"","video":""}  # 构造输出路径字典
    output_id = os.path.join(all_output_dir,dataset_name,video_name,str(target_id)) # 该目标的总输出路径
    if not no_video: output_dict["video"] = os.path.join(output_id, "video")  # 左bbox原图可视化右局部放大图拼接路径
    if not no_save_frames: output_dict["images"] = os.path.join(output_id, "images")  # 关键帧bbox可视化图像序列存放路径
    if not no_export_json: output_dict["json"] = os.path.join(output_id, "json")  # 关键帧标注存放.json文件的存储路径
    for path in output_dict.values():  # 路径构造
        if path: os.makedirs(path,exist_ok=True)
    return output_dict
