# 单目标处理入口
import argparse
from MOT_Trajectory_Tools.loader import *
from MOT_Trajectory_Tools.config import *
from MOT_Trajectory_Tools.extractor import *
from MOT_Trajectory_Tools.exporter import *
from MOT_Trajectory_Tools.visualizer import *

def make_parse():
    parser = argparse.ArgumentParser(description='MOT数据预处理（VideoLLAMA3专用版）')
    parser.add_argument('--dataset_entry', type=str, default="G:/UAVBenchmark", help='数据集总输入路径')
    parser.add_argument('--dataset_name',type=str,default='DynUAVI',help='数据集名称.可选：DynUAVI、VisDrone、UAVDT')
    parser.add_argument('--split_name',type=str,default='test',help="数据子集的名称")
    parser.add_argument('--video_name',type=str, default='009', help='视频名称')

    parser.add_argument('--target_id', type=int, default=14, help='要提取的目标ID')
    parser.add_argument('--start_frame', type=int, default=1, help='起始帧号（标注帧号）')
    parser.add_argument('--end_frame', type=int, default=1013, help='结束帧号（标注帧号）')

    parser.add_argument('--output_dir', type=str, default="G:/language-conditional_MOT", help='输出总目录')
    parser.add_argument('--no_video', action='store_true', help='不生成可视化视频')
    parser.add_argument('--no_save_frames', action='store_true', help='不生成关键帧上的bbox可视化')
    parser.add_argument('--no_export_json', action='store_true', help='不导出关键帧的bbox坐标')
    parser.add_argument('--no_export_videollama3_input', action='store_true', help='不导出videollama3模型的json输入字典')
    
    parser.add_argument('--add_key_frames', type=str, default='589,812,909,977',
                        help='补充关键帧列表，逗号分隔，如: 771,1204,1289,2028')
    
    args = parser.parse_args()
    return args


def main():
    # 1. 输入参数解析
    args = make_parse()  
    # 2. 获取图像序列文件夹路径和标注路径
    image_folder, annotation_path = get_image_anno_path(args.dataset_entry, 
            args.dataset_name, args.split_name, args.video_name)
    # 3. 从图像序列文件夹路径加载出所有的图像
    image_files = get_image_files(image_folder)
    # 4. 加载标注
    annotations = load_annotations(annotation_path)

    # 5. 构造输出目录
    output_dict = make_output_folders(args.output_dir, 
                  args.dataset_name, args.video_name, args.target_id, 
                  args.no_video, args.no_save_frames, args.no_export_json)
    
    # 6. 导出json标注
    if not args.no_export_json:
        export_bbox_json(args.start_frame, args.end_frame, annotations, 
                         args.target_id, output_dict["json"], SAMPLE_INTERVAL)
    
    # 7. 生成关键帧可视化图像序列
    if not args.no_save_frames:
        generate_visualization_key_images(args.start_frame,args.end_frame,args.add_key_frames,
                                        args.target_id, image_files,annotations,output_dict['images'])
        
    # 8. 生成轨迹视频
    if not args.no_video:
        video_path = generate_visualization_video(args.start_frame,args.end_frame,image_files,
                                     annotations,output_dict["video"],args.target_id,
                                     output_name=f"target_{args.target_id}_vis.mp4")
    
    # 9. 生成videollama3模型的输入字典
    if not args.no_export_videollama3_input and (not args.no_video):
        generate_videollama3_input(video_path,args.start_frame,args.end_frame,
                                   args.add_key_frames,output_dict["json"],args.target_id)

if __name__ == "__main__":
    main()


