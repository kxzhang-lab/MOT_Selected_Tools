# 批量处理入口
import argparse
from MOT_Trajectory_Tools.process_utils import *

def make_parse():
    parser = argparse.ArgumentParser(description='MOT数据预处理（VideoLLAMA3专用版）')
    parser.add_argument('--dataset_entry', type=str, default="G:/UAVBenchmark", help='数据集总输入路径')
    parser.add_argument('--dataset_name',type=str,default='DynUAVI',help='数据集名称.可选：DynUAVI、VisDrone、UAVDT')
    parser.add_argument('--split_name',type=str,default='test',help="数据子集的名称")
    parser.add_argument('--video_name',type=str, default='009', help='视频名称')

    # 新加测试参数：最大处理轨迹条数
    parser.add_argument('--max_num', type=int, default=None, help='最大处理轨迹条数')

    parser.add_argument('--output_dir', type=str, default="G:/language-conditional_MOT", help='输出总目录')
    parser.add_argument('--no_video', action='store_true', help='不生成可视化视频')
    parser.add_argument('--no_save_frames', action='store_true', help='不生成关键帧上的bbox可视化')
    parser.add_argument('--no_export_json', action='store_true', help='不导出关键帧的bbox坐标')
    parser.add_argument('--no_export_videollama3_input', action='store_true', help='不导出videollama3模型的json输入字典')
    
    parser.add_argument('--add_key_frames', type=str, default='',
                        help='补充关键帧列表，逗号分隔，如: 771,1204,1289,2028')
    
    args = parser.parse_args()
    return args


def make_target_param(target_id, target_anns, add_key_frames):
    """获取某条目标轨迹的输入参数字典
    Params:
        target_id: 目标轨迹序号
        target_anns: 该目标轨迹的所有标注
        add_key_frames: 人为指定的关键帧
    return:
        target_param (dict):{
            "target_id":目标轨迹ID
            "start_frame":起始时间戳
            "end_frame":结束时间戳
            "add_key_frames":人为补充的关键时间戳
        }
    """
    frames = list(target_anns.keys())
    frames.sort()
    start_frame, end_frame = frames[0], frames[-1]
    target_param = {"target_id":target_id, 
        "start_frame":start_frame, 
        "end_frame":end_frame, 
        "add_key_frames":add_key_frames
    }
    return target_param


def main():
    # 1. 输入参数解析
    args = make_parse()  
    # 2. 加载图像序列和视频标注信息
    image_files, annotations = load_images_annos(args.dataset_entry, 
            args.dataset_name, args.split_name, args.video_name)
    # 3. 构造输入参数
    input_param = {"output_dir":args.output_dir, "dataset_name":args.dataset_name, "video_name":args.video_name}
    save_flag = {"json":args.no_export_json,"images":args.no_save_frames, "video":args.no_video,"videollama3":args.no_export_videollama3_input}
    # 4. 最多处理的轨迹条数
    target_ids = list(annotations.keys())
    max_target_num = len(target_ids)
    if args.max_num: max_target_num = min(max_target_num,args.max_num)
    # 5. 针对每条轨迹依次处理
    for idx in range(max_target_num):
        target_param = make_target_param(target_ids[idx],annotations[target_ids[idx]],args.add_key_frames)
        target_conduct(input_param,target_param,save_flag,annotations,image_files)


if __name__ == "__main__":
    main()