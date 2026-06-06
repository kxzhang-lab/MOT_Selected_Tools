# 单目标处理入口
import argparse
from MOT_Trajectory_Tools.process_utils import *

def make_parse():
    parser = argparse.ArgumentParser(description='MOT数据预处理（VideoLLAMA3专用版）')
    parser.add_argument('--dataset_entry', type=str, default="G:/UAVBenchmark", help='数据集总输入路径')
    parser.add_argument('--dataset_name',type=str,default='DynUAVI',help='数据集名称.可选：DynUAVI、VisDrone、UAVDT')
    parser.add_argument('--split_name',type=str,default='test',help="数据子集的名称")
    parser.add_argument('--video_name',type=str, default='009', help='视频名称')

    parser.add_argument('--target_id', type=int, default=170, help='要提取的目标ID')
    parser.add_argument('--start_frame', type=int, default=1074, help='起始帧号（标注帧号）')
    parser.add_argument('--end_frame', type=int, default=1950, help='结束帧号（标注帧号）')

    parser.add_argument('--output_dir', type=str, default="G:/language-conditional_MOT", help='输出总目录')
    parser.add_argument('--no_video', action='store_true', help='不生成可视化视频')
    parser.add_argument('--no_save_frames', action='store_true', help='不生成关键帧上的bbox可视化')
    parser.add_argument('--no_export_json', action='store_true', help='不导出关键帧的bbox坐标')
    parser.add_argument('--no_export_videollama3_input', action='store_true', help='不导出videollama3模型的json输入字典')
    
    parser.add_argument('--add_key_frames', type=str, default='1554,1586,1667,1837',
                        help='补充关键帧列表，逗号分隔，如: 771,1204,1289,2028')
    
    args = parser.parse_args()
    return args


def main():
    # 1. 输入参数解析
    args = make_parse()  
    # 2. 加载图像序列和视频标注信息
    image_files, annotations = load_images_annos(args.dataset_entry, 
            args.dataset_name, args.split_name, args.video_name)
    # 3. 针对指定轨迹的处理
    input_param = {"output_dir":args.output_dir, "dataset_name":args.dataset_name, "video_name":args.video_name}
    target_param = {"target_id":args.target_id, "start_frame":args.start_frame, "end_frame":args.end_frame, "add_key_frames":args.add_key_frames}
    save_flag = {"json":args.no_export_json,"images":args.no_save_frames, "video":args.no_video,"videollama3":args.no_export_videollama3_input}
    target_conduct(input_param,target_param,save_flag,annotations,image_files)

if __name__ == "__main__":
    main()


