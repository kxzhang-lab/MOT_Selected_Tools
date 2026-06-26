# 批量处理工具函数
from MOT_Trajectory_Tools.loader import *
from MOT_Trajectory_Tools.config import *
from MOT_Trajectory_Tools.extractor import *
from MOT_Trajectory_Tools.exporter import *
from MOT_Trajectory_Tools.visualizer import *

def load_images_annos(dataset_entry,dataset_name,
                      split_name,video_name):
    """获取该视频的所有图像和标注。
    Params:
        dataset_entry: 数据集入口路径
        dataset_name: 数据集名称
        split_name: 子集名称
        video_name: 视频序号
    """ 
    # 1. 获取图像序列文件夹路径和标注路径
    image_folder, annotation_path = get_image_anno_path(dataset_entry, 
            dataset_name, split_name, video_name)
    # 2. 从图像序列文件夹路径加载出所有的图像
    image_files = get_image_files(image_folder)
    # 3. 加载标注
    annotations = load_annotations(annotation_path)
    return image_files, annotations


def target_conduct(input_param,target_param,save_flag,annotations,image_files):
    """单条轨迹的处理流程
    Params:
        input_param (dict):{
            "output_dir":所有数据集的总输出路径
            "dataset_name":数据集名称
            "video_name":视频名称
        }
        target_param (dict):{
            "target_id":目标轨迹ID
            "start_frame":起始时间戳
            "end_frame":结束时间戳
            "add_key_frames":人为补充的关键时间戳
        }
        save_flag (dict):{
            "json":是否导出标注的.json文件
            "images":是否导出关键时间戳下的bbox可视化效果图
            "video":是否导出拼接视频
            "videollama3":是否导出该模型的输入
        }
        annotations: 以轨迹ID为key的标注字典
        image_files: 所有图像序列
    """
    output_dir,dataset_name,video_name = input_param["output_dir"],\
        input_param["dataset_name"],input_param["video_name"]
    target_id,start_frame,end_frame,add_key_frames = \
        target_param["target_id"], target_param["start_frame"], \
            target_param["end_frame"], target_param["add_key_frames"]
    no_export_json,no_save_frames,no_video,no_export_videollama3_input = \
        save_flag["json"], save_flag["images"], save_flag["video"], save_flag["videollama3"]
    # 1. 构造输出目录
    output_dict = make_output_folders(output_dir, dataset_name, video_name, target_id,
                                        no_video, no_save_frames, no_export_json)
    
    # 2. 导出json标注
    if not no_export_json:
        export_bbox_json(start_frame, end_frame, annotations, target_id, output_dict["json"], SAMPLE_INTERVAL)
    
    # 3. 生成关键帧可视化图像序列
    if not no_save_frames:
        generate_visualization_key_images(start_frame, end_frame, add_key_frames, 
                                          target_id, image_files, annotations, output_dict['images'])
        
    # 4. 生成轨迹视频
    if not no_video:
        video_path = generate_visualization_video(start_frame,end_frame,image_files,
                                     annotations,output_dict["video"],target_id,
                                     output_name=f"target_{target_id}_vis.mp4")
    
    # 5. 生成videollama3模型的输入字典
    if not no_export_videollama3_input and (not no_video):
        generate_videollama3_input(video_path,start_frame,end_frame,
                                   add_key_frames,output_dict["json"],target_id)